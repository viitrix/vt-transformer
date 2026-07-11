// radix_cache_test.cpp — Host-only tests for RadixCache<>
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I .. radix_cache_test.cpp -o /tmp/radix_test
// Run:
//   /tmp/radix_test

#include <core/radix.hpp>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using minisgl::RadixCache;
using minisgl::HostAllocator;

static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << (msg) << std::endl;                                \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                 \
    do {                                                                    \
        auto _a = (a);                                                      \
        auto _b = (b);                                                      \
        if (!(_a == _b)) {                                                  \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << (msg) << " (got " << _a << ", want " << _b << ")"  \
                      << std::endl;                                         \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

using TokenVec = std::vector<int32_t>;
using IndexVec = std::vector<int32_t>;

// ============================================================================
// Tests with page_size = 1
// ============================================================================

void test_empty_match() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    auto m = cache.match_prefix({1, 2, 3});
    CHECK_EQ(m.handle.cached_len, 0u, "empty cache should return 0");
    CHECK(m.handle.node == cache.root(), "should match root");
    cache.check_integrity();
}

void test_basic_insert_then_match() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    TokenVec ids = {1, 2, 3, 4, 5};
    IndexVec idx = {10, 11, 12, 13, 14};
    auto ins = cache.insert_prefix(ids, idx);
    CHECK_EQ(ins.cached_len, 0u, "first insert: nothing was cached");
    CHECK_EQ(ins.handle.cached_len, ids.size(), "insert_len should equal ids size");

    auto m = cache.match_prefix(ids);
    CHECK_EQ(m.handle.cached_len, ids.size(), "full match expected");

    auto gathered = cache.get_matched_indices(m.handle).to_host();
    CHECK(gathered == idx, "gathered indices should match inserted indices");
    cache.check_integrity();
}

void test_partial_match_triggers_split() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2, 3, 4, 5}, {100, 101, 102, 103, 104});

    // Partially-matching second prefix should split the existing node.
    auto m = cache.match_prefix({1, 2, 9, 9, 9});
    CHECK_EQ(m.handle.cached_len, 2u, "should match first 2 tokens");
    cache.check_integrity();

    // Both prefixes should still resolve to the right lengths.
    auto m1 = cache.match_prefix({1, 2, 3, 4, 5});
    CHECK_EQ(m1.handle.cached_len, 5u, "first prefix should still fully match");
    auto m2 = cache.match_prefix({1, 2, 9, 9, 9});
    CHECK_EQ(m2.handle.cached_len, 2u, "second prefix should match the 2-token prefix");
}

void test_extend_with_child() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2, 3}, {10, 11, 12});

    // Extending an existing prefix should add a new child node, not split.
    cache.insert_prefix({1, 2, 3, 4, 5, 6}, {10, 11, 12, 13, 14, 15});

    auto m = cache.match_prefix({1, 2, 3, 4, 5, 6});
    CHECK_EQ(m.handle.cached_len, 6u, "extended prefix should fully match");
    cache.check_integrity();
}

void test_lock_blocks_evict() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2, 3}, {10, 11, 12});
    cache.insert_prefix({4, 5, 6}, {20, 21, 22});

    auto m = cache.match_prefix({1, 2, 3});
    cache.lock_handle(m.handle);              // protect first segment
    CHECK_EQ(cache.size_info().protected_size, 3u, "protected size after lock");

    // Evicting must skip the protected segment and pick the other one.
    auto ev = cache.evict(3).to_host();
    CHECK_EQ(ev.size(), 3u, "should evict exactly 3 tokens");
    CHECK_EQ(cache.size_info().evictable_size, 0u, "evictable should be drained");
    CHECK_EQ(cache.size_info().protected_size, 3u, "protected should be unchanged");

    // After unlock, the previously-protected segment becomes evictable.
    cache.lock_handle(m.handle, /*unlock=*/true);
    CHECK_EQ(cache.size_info().evictable_size, 3u, "evictable after unlock");
    CHECK_EQ(cache.size_info().protected_size, 0u, "protected after unlock");
    cache.check_integrity();
}

void test_evict_over_budget_throws() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2, 3}, {10, 11, 12});
    bool threw = false;
    try {
        cache.evict(999);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "evict over budget must throw");
}

void test_lru_picks_oldest_first() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2}, {10, 11});      // oldest
    cache.insert_prefix({3, 4}, {20, 21});      // newer
    cache.insert_prefix({5, 6}, {30, 31});      // newest

    auto ev = cache.evict(2).to_host();
    CHECK_EQ(ev.size(), 2u, "evicted batch size");
    CHECK(ev == IndexVec({10, 11}),
          "oldest segment (1,2) should be evicted first");
    cache.check_integrity();
}

// ============================================================================
// Tests with page_size = 4
// ============================================================================

void test_page_align_truncates_tail() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/4);
    // 9 tokens → insert_len should align_down to 8.
    TokenVec ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    IndexVec idx = {100, 101, 102, 103, 104, 105, 106, 107, 108};
    auto ins = cache.insert_prefix(ids, idx);
    CHECK_EQ(ins.handle.cached_len, 8u, "insert_len should be align_down(9,4)=8");
    CHECK_EQ(cache.size_info().evictable_size, 8u, "evictable should reflect 8 tokens");

    // Matching the full 9-token input should only return 8.
    auto m = cache.match_prefix(ids);
    CHECK_EQ(m.handle.cached_len, 8u, "match should be page-aligned");
    cache.check_integrity();
}

void test_page_size_4_partial_match_within_page() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/4);
    cache.insert_prefix({1, 2, 3, 4, 5, 6, 7, 8}, {10, 11, 12, 13, 14, 15, 16, 17});

    // First 5 tokens match (1 token into page 2); aligned_down → 4.
    auto m = cache.match_prefix({1, 2, 3, 4, 5, 9, 9, 9});
    CHECK_EQ(m.handle.cached_len, 4u, "match should align_down to 4");
    cache.check_integrity();
}

void test_page_size_4_split_at_page_boundary() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/4);
    cache.insert_prefix({1, 2, 3, 4, 5, 6, 7, 8},
                        {10, 11, 12, 13, 14, 15, 16, 17});

    // Divergence in page 3 should align_down to 8 — no split.
    auto m = cache.match_prefix({1, 2, 3, 4, 5, 6, 7, 8, 9, 9});
    CHECK_EQ(m.handle.cached_len, 8u, "first 8 should match cleanly");

    // Divergence in page 2 should align_down to 4 — splits the node at 4.
    auto m2 = cache.match_prefix({1, 2, 3, 4, 5, 6, 9, 9});
    CHECK_EQ(m2.handle.cached_len, 4u, "should match only first page");
    cache.check_integrity();
}

void test_get_matched_indices_multi_hop() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    cache.insert_prefix({1, 2, 3},       {10, 11, 12});        // first leaf
    cache.insert_prefix({1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}); // extend

    auto m = cache.match_prefix({1, 2, 3, 4, 5});
    auto v = cache.get_matched_indices(m.handle).to_host();
    CHECK(v == IndexVec({10, 11, 12, 13, 14}),
          "gathered indices should concatenate parent → child");
}

void test_reset_throws() {
    RadixCache<int32_t, int32_t, HostAllocator> cache(/*page_size=*/1);
    bool threw = false;
    try {
        cache.reset();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "reset must throw (parity with Python)");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        test_empty_match,
        test_basic_insert_then_match,
        test_partial_match_triggers_split,
        test_extend_with_child,
        test_lock_blocks_evict,
        test_evict_over_budget_throws,
        test_lru_picks_oldest_first,
        test_page_align_truncates_tail,
        test_page_size_4_partial_match_within_page,
        test_page_size_4_split_at_page_boundary,
        test_get_matched_indices_multi_hop,
        test_reset_throws,
    };

    for (TestFn t : tests) {
        t();
    }

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED (" << sizeof(tests)/sizeof(tests[0]) << ")\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)\n";
    return 1;
}
