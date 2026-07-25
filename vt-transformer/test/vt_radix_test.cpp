// vt_radix_test.cpp — Tests for core/vt_radix.hpp (RadixTree)
//
// vt_radix.hpp is header-only; no .cpp linkage needed.
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I.. vt_radix_test.cpp -o /tmp/vt_radix_test
// Run:
//   /tmp/vt_radix_test

#include <core/vt_radix.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using Tree = vt::RadixTree<int32_t, int32_t>;
using Node = vt::RadixNode<int32_t, int32_t>;

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

// Compare two index vectors as sets (evict order isn't part of the contract).
static bool same_set(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    auto sa = a; std::sort(sa.begin(), sa.end());
    auto sb = b; std::sort(sb.begin(), sb.end());
    return sa == sb;
}

// ============================================================================
// (1) Query: match_prefix  — must be read-only
// ============================================================================

void test_match_empty_tree() {
    Tree t(/*page_size=*/1);
    auto r = t.match_prefix({1, 2, 3});
    CHECK_EQ(r.prefix_len, 0u, "empty tree prefix_len");
    CHECK(r.node == t.root(), "empty tree lands at root");
}

void test_match_full() {
    Tree t;
    t.insert_prefix({1, 2, 3, 4}, {10, 20, 30, 40});
    auto r = t.match_prefix({1, 2, 3, 4});
    CHECK_EQ(r.prefix_len, 4u, "exact full match");
}

void test_match_partial_inside_node() {
    // page_size=1, insert [1,2,3,4] as a single node, then query [1,2,9,9].
    // Tokens diverge at offset 2 → partial match reported (length 2), no split.
    Tree t(/*page_size=*/1);
    t.insert_prefix({1, 2, 3, 4}, {10, 20, 30, 40});
    auto r = t.match_prefix({1, 2, 9, 9});
    CHECK_EQ(r.prefix_len, 2u, "partial match at token 3");
    CHECK_EQ(t.total_size(), 4u, "Query must not split the node");
}

void test_match_is_readonly() {
    // Query must not perturb stats.
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    const size_t ev_before = t.evictable_size();
    const size_t pr_before = t.protected_size();
    auto r = t.match_prefix({1, 2, 3});
    (void)r;
    CHECK_EQ(t.evictable_size(), ev_before, "Query leaves evictable_size alone");
    CHECK_EQ(t.protected_size(), pr_before, "Query leaves protected_size alone");
}

// ============================================================================
// (2) Insert: insert_prefix  — sole owner of structural mutation
// ============================================================================

void test_insert_simple() {
    Tree t;
    auto r = t.insert_prefix({1, 2, 3}, {10, 20, 30});
    CHECK_EQ(r.prefix_len,    0u, "no prior prefix");
    CHECK_EQ(r.inserted_len,  3u, "all 3 tokens written");
    CHECK_EQ(t.total_size(),  3u, "total grows by 3");
    CHECK_EQ(t.evictable_size(), 3u, "fresh tokens are evictable");
    CHECK_EQ(t.protected_size(), 0u, "nothing locked yet");
}

void test_insert_with_split() {
    // page_size=1: insert [1,2,3,4], then [1,2,9,9] — must split at [1,2].
    Tree t(/*page_size=*/1);
    t.insert_prefix({1, 2, 3, 4}, {10, 20, 30, 40});
    auto r = t.insert_prefix({1, 2, 9, 9}, {50, 60, 70, 80});
    CHECK_EQ(r.prefix_len,   2u, "shared prefix length");
    CHECK_EQ(r.inserted_len, 4u, "page-aligned insert length");
    // head[1,2] (2) + tail[3,4] (2) + new[9,9] (2) = 6
    CHECK_EQ(t.total_size(), 6u, "split is size-neutral; +2 for the new child");
}

void test_insert_with_split_larger_page() {
    // page_size=2: insert [1,2,3,4], then [1,2,5,6] — splits at [1,2].
    Tree t(/*page_size=*/2);
    t.insert_prefix({1, 2, 3, 4}, {10, 20, 30, 40});
    t.insert_prefix({1, 2, 5, 6}, {50, 60, 70, 80});
    // head[1,2] (2) + A[3,4] (2) + B[5,6] (2) = 6
    CHECK_EQ(t.total_size(), 6u, "page_size=2 split layout");
}

void test_insert_no_op_when_fully_present() {
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    auto r = t.insert_prefix({1, 2, 3}, {99, 98, 97});   // ignored payload
    CHECK_EQ(r.prefix_len,   3u, "fully present");
    CHECK_EQ(r.inserted_len, 3u, "no partial");
    CHECK_EQ(t.total_size(), 3u, "size unchanged");
    // Original indices survive (we evict and check what comes back).
    auto freed = t.evict(3);
    CHECK(same_set(freed, {10, 20, 30}), "original indices preserved");
}

void test_insert_extends_existing_chain() {
    // Insert prefix, then extend with a suffix → second insert attaches a child.
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    auto r = t.insert_prefix({1, 2, 3, 4, 5}, {0, 0, 0, 40, 50});
    CHECK_EQ(r.prefix_len,   3u, "first 3 already in tree");
    CHECK_EQ(r.inserted_len, 5u, "page-aligned length");
    CHECK_EQ(t.total_size(), 5u, "only +2 for the new suffix");
}

void test_insert_below_page_size() {
    Tree t(/*page_size=*/4);
    auto r = t.insert_prefix({1, 2, 3}, {10, 20, 30});   // < page_size
    CHECK_EQ(r.inserted_len, 0u, "below page_size: nothing to insert");
    CHECK_EQ(t.total_size(), 0u, "tree still empty");
}

void test_insert_page_aligned_down() {
    Tree t(/*page_size=*/4);
    auto r = t.insert_prefix({1, 2, 3, 4, 5, 6}, {10, 20, 30, 40, 50, 60});
    CHECK_EQ(r.inserted_len, 4u, "align_down(6,4) = 4");
    CHECK_EQ(t.total_size(), 4u, "only 4 tokens actually stored");
}

// ============================================================================
// (3) Lock / Unlock  — only ref_count
// ============================================================================

void test_lock_unlock_basic() {
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    Node* n = t.match_prefix({1, 2, 3}).node;

    CHECK_EQ(t.protected_size(), 0u, "nothing protected before lock");
    t.lock(n);
    CHECK_EQ(t.protected_size(), 3u, "all 3 tokens protected");
    CHECK_EQ(t.evictable_size(), 0u, "evictable shrinks");
    t.unlock(n);
    CHECK_EQ(t.protected_size(), 0u, "protected back to 0");
    CHECK_EQ(t.evictable_size(), 3u, "evictable restored");
}

void test_lock_propagates_to_parent() {
    // Two-level tree: lock deep leaf should lock the whole path to root.
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});              // A: [1,2,3]
    t.insert_prefix({1, 2, 3, 4, 5}, {0, 0, 0, 40, 50});   // B: [4,5] under A
    Node* leaf = t.match_prefix({1, 2, 3, 4, 5}).node;
    CHECK(leaf != t.root(), "got a non-root leaf");

    t.lock(leaf);
    CHECK_EQ(t.protected_size(), 5u, "A(3) + B(2) both locked");
    t.unlock(leaf);
    CHECK_EQ(t.protected_size(), 0u, "fully unlocked");
}

void test_lock_idempotent_for_size() {
    // Locking twice bumps ref_count to 2 but protected_size only counts once.
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    Node* n = t.match_prefix({1, 2, 3}).node;

    t.lock(n);
    t.lock(n);
    CHECK_EQ(t.protected_size(), 3u, "size counts tokens, not ref_count");
    t.unlock(n);
    CHECK_EQ(t.protected_size(), 3u, "still protected (ref_count=1)");
    t.unlock(n);
    CHECK_EQ(t.protected_size(), 0u, "fully unlocked");
}

// ============================================================================
// (4) Evict  — only removes leaves, returns slot indices
// ============================================================================

void test_evict_returns_freed_indices() {
    Tree t;
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    auto freed = t.evict(3);
    CHECK(same_set(freed, {10, 20, 30}), "all slot indices returned");
    CHECK_EQ(t.total_size(), 0u, "tree empty after evict");
}

void test_evict_oldest_first() {
    Tree t;
    t.insert_prefix({1, 2}, {10, 20});   // older timestamp
    t.insert_prefix({3, 4}, {30, 40});   // newer timestamp
    auto freed = t.evict(2);
    CHECK(same_set(freed, {10, 20}), "older ([1,2]) evicted first");
    CHECK_EQ(t.total_size(), 2u, "newer remains");
}

void test_evict_skips_locked() {
    Tree t;
    t.insert_prefix({1, 2}, {10, 20});
    t.insert_prefix({3, 4}, {30, 40});
    Node* older = t.match_prefix({1, 2}).node;
    t.lock(older);                       // protect the older node
    auto freed = t.evict(2);
    CHECK(same_set(freed, {30, 40}), "only the unlocked leaf is fair game");
    CHECK_EQ(t.evictable_size(), 0u, "locked node not evictable");
    t.unlock(older);
}

void test_evict_promotes_parent_to_leaf() {
    // Build head[1,2] → {A:[3], B:[9]} (page_size=1, lengths: 2+1+1=4).
    Tree t(/*page_size=*/1);
    t.insert_prefix({1, 2, 3}, {10, 20, 30});
    t.insert_prefix({1, 2, 9}, {40, 50, 60});
    CHECK_EQ(t.total_size(), 4u, "tree setup");

    auto freed = t.evict(2);             // drops A + B (1+1 = 2)
    CHECK(same_set(freed, {30, 60}), "both children's slots freed");
    CHECK_EQ(t.evictable_size(), 2u, "parent became an evictable leaf");

    freed = t.evict(2);                  // now drops head
    CHECK(same_set(freed, {10, 20}), "parent's slots freed");
    CHECK_EQ(t.total_size(), 0u, "tree empty");
}

// ============================================================================
// Stats invariant: total == evictable + protected, across mixed ops
// ============================================================================

void test_stats_invariant() {
    Tree t(/*page_size=*/2);
    t.insert_prefix({1, 2, 3, 4}, {10, 20, 30, 40});
    t.insert_prefix({1, 2, 5, 6}, {50, 60, 70, 80});

    Node* n = t.match_prefix({1, 2, 3, 4}).node;
    t.lock(n);
    CHECK_EQ(t.total_size(), t.evictable_size() + t.protected_size(),
             "invariant holds after lock");
    t.unlock(n);
    CHECK_EQ(t.total_size(), t.evictable_size() + t.protected_size(),
             "invariant holds after unlock");
    t.evict(2);
    CHECK_EQ(t.total_size(), t.evictable_size() + t.protected_size(),
             "invariant holds after evict");
}

// ============================================================================
// main
// ============================================================================

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        // Query
        test_match_empty_tree,
        test_match_full,
        test_match_partial_inside_node,
        test_match_is_readonly,
        // Insert
        test_insert_simple,
        test_insert_with_split,
        test_insert_with_split_larger_page,
        test_insert_no_op_when_fully_present,
        test_insert_extends_existing_chain,
        test_insert_below_page_size,
        test_insert_page_aligned_down,
        // Lock
        test_lock_unlock_basic,
        test_lock_propagates_to_parent,
        test_lock_idempotent_for_size,
        // Evict
        test_evict_returns_freed_indices,
        test_evict_oldest_first,
        test_evict_skips_locked,
        test_evict_promotes_parent_to_leaf,
        // Stats
        test_stats_invariant,
    };

    for (TestFn fn : tests) fn();

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED (" << sizeof(tests)/sizeof(tests[0]) << ")\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)\n";
    return 1;
}
