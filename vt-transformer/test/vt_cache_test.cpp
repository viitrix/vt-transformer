// vt_cache_test.cpp — Tests for core/vt_cache.hpp (CacheManager)
//
// Covers page_size=1 (regression) and page_size=2 (page-aligned page_ids).
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I.. vt_cache_test.cpp -o /tmp/vt_cache_test
// Run:
//   /tmp/vt_cache_test

#include <core/vt_cache.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using CM = vt::CacheManager<int32_t, int32_t>;

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

static bool all_aligned(const std::vector<int32_t>& v, int page_size) {
    for (int32_t x : v) if (x % page_size != 0) return false;
    return true;
}

// ============================================================================
// page_size=1 — basic prepare / allocate / finished round-trip
// ============================================================================

void test_basic_round_trip_page_size_1() {
    CM cm(/*max_running_reqs=*/2, /*num_pages=*/8, /*max_seq_len=*/8, /*page_size=*/1);
    auto* pt = cm.pt();
    CHECK_EQ(pt->page_size(), 1, "pt->page_size()");
    CHECK_EQ(cm.available_size(), 8u, "empty tree: 8 pages × 1 = 8 tokens");

    int32_t table_idx = pt->alloc_row();
    CHECK_EQ(table_idx, 0, "first alloc_row");

    std::vector<int32_t> input = {1, 2, 3};
    auto pr = cm.prepare(table_idx, input);
    CHECK_EQ(pr.cached_len, 0, "empty tree prefix_len");
    CHECK(pr.node != nullptr, "node not null");

    bool ok = cm.allocate_pages({{table_idx, /*first_pos=*/0, /*last_pos=*/3}});
    CHECK(ok, "allocate_pages should succeed");
    CHECK_EQ(pt->available_pages(), 5, "8 - 3 = 5 pages left");

    // LIFO pops lowest first (constructor pushes high→low, alloc pops low first):
    // page_ids come out as 0, 1, 2 for page_size=1.
    const int32_t* row = pt->page_row(table_idx);
    CHECK_EQ(row[0], 0, "row[0] = 0");
    CHECK_EQ(row[1], 1, "row[1] = 1");
    CHECK_EQ(row[2], 2, "row[2] = 2");

    // finished: insert into radix. The 3 pages now belong to radix (NOT freed),
    // because they're the newly-inserted range [prefix_len, inserted_len).
    // Only the row is returned.
    CM::FinishInput fi{
        table_idx, pr, /*cur_cached_len=*/3,
        /*tokens=*/std::vector<int32_t>{1, 2, 3},
    };
    cm.finished(fi);
    CHECK_EQ(pt->available_pages(), 5, "pages stay in radix after finished");
    CHECK_EQ(pt->available_rows(), 2, "row returned after finished");
}

// ============================================================================
// page_size=2 — same round-trip, verify broadcast + aligned ids
// ============================================================================

void test_basic_round_trip_page_size_2() {
    CM cm(/*max_running_reqs=*/2, /*num_pages=*/8, /*max_seq_len=*/8, /*page_size=*/2);
    auto* pt = cm.pt();
    CHECK_EQ(pt->page_size(), 2, "pt->page_size()");

    int32_t table_idx = pt->alloc_row();
    std::vector<int32_t> input = {1, 2, 3, 4};  // 4 tokens = 2 pages
    auto pr = cm.prepare(table_idx, input);
    CHECK_EQ(pr.cached_len, 0, "empty tree");

    bool ok = cm.allocate_pages({{table_idx, 0, 4}});
    CHECK(ok, "allocate_pages");
    CHECK_EQ(pt->available_pages(), 6, "8 - 2 = 6 pages left (page_size=2)");

    // LIFO: page_ids come out as 0 then 2 (page-aligned).
    // Each broadcast within its page_size=2 slot window.
    const int32_t* row = pt->page_row(table_idx);
    CHECK_EQ(row[0], 0, "row[0] = 0 (first allocated page_id)");
    CHECK_EQ(row[1], 0, "row[1] = 0 (broadcast)");
    CHECK_EQ(row[2], 2, "row[2] = 2 (second allocated page_id)");
    CHECK_EQ(row[3], 2, "row[3] = 2 (broadcast)");

    CM::FinishInput fi{
        table_idx, pr, /*cur_cached_len=*/4,
        /*tokens=*/std::vector<int32_t>{1, 2, 3, 4},
    };
    cm.finished(fi);
    CHECK_EQ(pt->available_pages(), 6, "2 pages stay in radix");
}

// ============================================================================
// Prefix reuse — finished inserts prefix, second prepare hits it
// ============================================================================

void test_prefix_reuse_page_size_2() {
    CM cm(2, 8, 8, /*page_size=*/2);
    auto* pt = cm.pt();

    // First req: insert [1,2,3,4] → page_ids [0, 2]
    int32_t t1 = pt->alloc_row();
    auto pr1 = cm.prepare(t1, {1, 2, 3, 4});
    CHECK_EQ(pr1.cached_len, 0, "first req empty hit");
    cm.allocate_pages({{t1, 0, 4}});
    CM::FinishInput fi1{t1, pr1, 4, {1, 2, 3, 4}};
    cm.finished(fi1);
    CHECK_EQ(pt->available_pages(), 6, "2 pages now radix-owned");

    // Second req: prepare [1,2,3,4,5,6] — should hit 4-token prefix
    int32_t t2 = pt->alloc_row();
    auto pr2 = cm.prepare(t2, {1, 2, 3, 4, 5, 6});
    CHECK_EQ(pr2.cached_len, 4, "second req hits 4-token prefix");

    // Reused page_ids should match what t1 had (broadcast within page).
    const int32_t* row2 = pt->page_row(t2);
    CHECK_EQ(row2[0], 0, "reused row2[0] = 0");
    CHECK_EQ(row2[1], 0, "reused row2[1] = 0");
    CHECK_EQ(row2[2], 2, "reused row2[2] = 2");
    CHECK_EQ(row2[3], 2, "reused row2[3] = 2");

    // Extend: allocate_pages for [4, 6) → 1 new page (page_id 4, next LIFO pop)
    bool ok = cm.allocate_pages({{t2, 4, 6}});
    CHECK(ok, "allocate_pages for tail");
    CHECK_EQ(pt->available_pages(), 5, "8 - 2 (t1) - 1 (new) = 5");
    CHECK_EQ(row2[4], 4, "row2[4] = 4 (new page_id)");
    CHECK_EQ(row2[5], 4, "row2[5] = 4 (broadcast)");

    CM::FinishInput fi2{t2, pr2, 6, {1, 2, 3, 4, 5, 6}};
    cm.finished(fi2);
}

// ============================================================================
// allocate_pages skips already-allocated pages (decode into partial page)
// ============================================================================

void test_allocate_skips_partial_page() {
    CM cm(2, 8, 16, /*page_size=*/4);
    auto* pt = cm.pt();
    int32_t t = pt->alloc_row();
    auto pr = cm.prepare(t, {1, 2, 3, 4, 5, 6});  // 6 tokens
    CHECK_EQ(pr.cached_len, 0, "empty tree");

    // First alloc: [0, 6) → ceil(6/4) = 2 pages. page_ids 0 (page 0) and 4 (page 1).
    // Broadcast writes the FULL extent of each page (matches mini-sglang: partial tail
    // is over-allocated; attention kernel only reads cached_len tokens).
    bool ok = cm.allocate_pages({{t, 0, 6}});
    CHECK(ok, "first allocate");
    CHECK_EQ(pt->available_pages(), 6, "8 - 2 = 6");

    const int32_t* row = pt->page_row(t);
    int32_t page0 = row[0];  // covers [0..3]
    int32_t page1 = row[4];  // covers [4..7] (kernel uses only [4,5])
    CHECK_EQ(page0, 0, "page 0 id");
    CHECK_EQ(page1, 4, "page 1 id");
    CHECK_EQ(row[3], page0, "page 0 broadcast");
    CHECK_EQ(row[4], page1, "page 1 start");
    CHECK_EQ(row[5], page1, "page 1 mid");
    CHECK_EQ(row[6], page1, "page 1 over-alloc slot 6 (broadcast writes full page)");
    CHECK_EQ(row[7], page1, "page 1 over-alloc slot 7 (broadcast writes full page)");

    // Second alloc: extend [6, 9). first_pos=6 is mid-page-1 (already allocated).
    // Should skip page 1 entirely and only allocate page 2 (covers [8..11]).
    ok = cm.allocate_pages({{t, 6, 9}});
    CHECK(ok, "second allocate (with skip)");
    CHECK_EQ(pt->available_pages(), 5, "only 1 new page allocated (page 1 reused)");

    // page 1 untouched
    CHECK_EQ(row[4], page1, "page 1 still has same id");
    CHECK_EQ(row[5], page1, "page 1 still has same id");
    // page 2 newly allocated (page_id 8), broadcast to [8..11]
    CHECK_EQ(row[8], 8, "page 2 slot 8 = 8");
    CHECK_EQ(row[9], 8, "page 2 slot 9 = 8");
    CHECK_EQ(row[10], 8, "page 2 slot 10 = 8");
    CHECK_EQ(row[11], 8, "page 2 slot 11 = 8");
}

// ============================================================================
// evict returns per-page deduped, page-aligned ids (not per-token replicates)
// ============================================================================

void test_evict_returns_unique_page_ids() {
    CM cm(2, 8, 8, /*page_size=*/2);
    auto* pt = cm.pt();

    // Insert [1,2,3,4] (2 pages) into radix via finished
    int32_t t = pt->alloc_row();
    auto pr = cm.prepare(t, {1, 2, 3, 4});
    cm.allocate_pages({{t, 0, 4}});
    CM::FinishInput fi{t, pr, 4, {1, 2, 3, 4}};
    cm.finished(fi);
    // After finished, radix value_ = [0,0,2,2] (per-token, broadcast within page).
    // evict(4) should dedup → [0, 2], not [0,0,2,2].

    auto freed = cm.evict(4);
    CHECK_EQ(freed.size(), 2u, "evict returns per-page count, not per-token");
    CHECK(all_aligned(freed, 2), "all returned ids are page-aligned");
    CHECK(freed[0] != freed[1], "two distinct page_ids");

    // Caller can safely free each id once
    pt->free_pages(freed.data(), (int)freed.size());
    CHECK_EQ(pt->available_pages(), 8, "pool back to full after free");
}

void test_evict_too_greedy_returns_empty() {
    CM cm(2, 8, 8, /*page_size=*/2);
    auto* pt = cm.pt();
    int32_t t = pt->alloc_row();
    auto pr = cm.prepare(t, {1, 2, 3, 4});
    cm.allocate_pages({{t, 0, 4}});
    CM::FinishInput fi{t, pr, 4, {1, 2, 3, 4}};
    cm.finished(fi);

    // Only 4 evictable tokens; ask for 6 → empty (no partial evict)
    auto freed = cm.evict(6);
    CHECK(freed.empty(), "over-budget evict returns empty");
}

// ============================================================================
// available_size is in tokens (page_size multiplied in)
// ============================================================================

void test_available_size_units() {
    CM cm(2, 8, 8, /*page_size=*/2);
    auto* pt = cm.pt();
    // Empty tree: 8 pages × 2 tokens/page = 16 tokens worth of room
    CHECK_EQ(cm.available_size(), 16u, "8 pages × page_size 2 = 16 tokens");

    int32_t t = pt->alloc_row();
    auto pr = cm.prepare(t, {1, 2, 3, 4});
    cm.allocate_pages({{t, 0, 4}});
    // 2 pages consumed → 6 pages × 2 = 12 tokens room (radix still empty)
    CHECK_EQ(cm.available_size(), 12u, "after alloc: 6 pages × 2 = 12");

    CM::FinishInput fi{t, pr, 4, {1, 2, 3, 4}};
    cm.finished(fi);
    // After finished: radix owns 4 tokens (evictable), pool still has 6 pages.
    // available_size = 4 (radix evictable) + 12 (pool) = 16.
    // Same as initial — radix prefix is reusable, so it counts as available.
    CHECK_EQ(cm.available_size(), 16u, "after finished: 4 radix + 12 pool = 16");
}

// ============================================================================
// main
// ============================================================================

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        test_basic_round_trip_page_size_1,
        test_basic_round_trip_page_size_2,
        test_prefix_reuse_page_size_2,
        test_allocate_skips_partial_page,
        test_evict_returns_unique_page_ids,
        test_evict_too_greedy_returns_empty,
        test_available_size_units,
    };

    for (TestFn fn : tests) fn();

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED (" << sizeof(tests)/sizeof(tests[0]) << ")\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)\n";
    return 1;
}
