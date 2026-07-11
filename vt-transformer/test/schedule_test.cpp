// schedule_test.cpp — 组件级测试
//
// 测 PrefillManager / DecodeManager 等可独立验证的组件,不走 ZMQ、不开线程。
// 完整的端到端测试需要 ZMQ + msgpack 编码 UserMsg,首版先跳过。
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I .. schedule_test.cpp -o /tmp/schedule_test
// Run:
//   /tmp/schedule_test

#include <core/cache.hpp>
#include <core/config.hpp>
#include <core/decode.hpp>
#include <core/prefill.hpp>
#include <core/req.hpp>
#include <core/table.hpp>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using minisgl::Batch;
using minisgl::BatchPhase;
using minisgl::CacheManager;
using minisgl::DecodeManager;
using minisgl::HostAllocator;
using minisgl::PendingReq;
using minisgl::PrefillManager;
using minisgl::Req;
using minisgl::SamplingParams;
using minisgl::SchedulerConfig;
using minisgl::TableManager;
using minisgl::UserMsg;

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
        auto _a = (a);                                                       \
        auto _b = (b);                                                       \
        if (!(_a == _b)) {                                                   \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << (msg) << " (got " << _a << ", want " << _b << ")"  \
                      << std::endl;                                          \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

// 测试夹具:装一套 cache/table/decode/prefill 共享的依赖。
struct Fixture {
    static constexpr size_t PAGE_SIZE = 1;
    static constexpr size_t NUM_PAGES = 1024;
    static constexpr size_t MAX_REQS  = 64;
    static constexpr size_t MAX_SEQ_LEN = 32;

    CacheManager<int32_t, int32_t, HostAllocator> cm{PAGE_SIZE, NUM_PAGES};
    TableManager                                  tm{MAX_REQS};
    DecodeManager                                 dm{PAGE_SIZE};
    std::unordered_map<int64_t, std::shared_ptr<Req>> live_reqs;

    std::unique_ptr<PrefillManager> pm;

    Fixture() {
        pm = std::make_unique<PrefillManager>(cm, tm, dm, live_reqs, MAX_SEQ_LEN);
    }

    UserMsg make_user_msg(int64_t uid, std::vector<int32_t> ids,
                          int64_t max_tokens = 4) const {
        UserMsg m;
        m.uid             = uid;
        m.input_ids       = std::move(ids);
        m.sampling_params = SamplingParams{};
        m.sampling_params.max_tokens = max_tokens;
        return m;
    }
};

// ============================================================================
// PrefillManager
// ============================================================================

void test_prefill_admit_one_req() {
    Fixture f;
    CHECK(f.pm->pending_size() == 0u, "starts empty");
    CHECK(!f.pm->runnable(), "not runnable when empty");

    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3, 4}));
    CHECK(f.pm->runnable(), "runnable after add");
    CHECK_EQ(f.pm->pending_size(), 1u, "one pending");

    auto batch = f.pm->schedule_next_batch(/*prefill_budget=*/8192);
    CHECK(batch != nullptr, "should produce a batch");
    CHECK(batch->is_prefill(), "phase is prefill");
    CHECK_EQ(batch->reqs.size(), 1u, "one req admitted");
    CHECK_EQ(f.pm->pending_size(), 0u, "pending drained");

    auto& r = batch->reqs[0];
    CHECK_EQ(r->uid, 1, "uid preserved");
    CHECK_EQ(r->cached_len, 0u, "no cache hit on first req");
    CHECK_EQ(r->device_len, 4u, "device_len == input_len");
    CHECK_EQ(r->max_device_len, 8u, "input_len + output_len");
    CHECK(r->table_idx >= 0, "table_idx allocated");
    CHECK(f.live_reqs.count(1) == 1u, "req in live_reqs");
    f.cm.check_integrity();
}

void test_prefill_token_budget_limit() {
    Fixture f;
    // 三个长 prompt,各 10 token;budget=15 应只够一个完整 prefill
    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    f.pm->add_one_req(f.make_user_msg(2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    f.pm->add_one_req(f.make_user_msg(3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    auto batch = f.pm->schedule_next_batch(/*prefill_budget=*/15);
    CHECK(batch != nullptr, "first batch should admit one");
    CHECK_EQ(batch->reqs.size(), 1u, "only one req fits in budget");
    CHECK_EQ(f.pm->pending_size(), 2u, "two still pending");
    f.cm.check_integrity();
}

void test_prefill_cache_available_size_limit() {
    Fixture f;
    // cache 太小,大 prompt 装不下,应该拒绝
    f.cm = CacheManager<int32_t, int32_t, HostAllocator>{/*page_size=*/1, /*num_pages=*/4};
    f.pm = std::make_unique<PrefillManager>(f.cm, f.tm, f.dm, f.live_reqs, Fixture::MAX_SEQ_LEN);

    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3, 4, 5, 6, 7, 8}));
    auto batch = f.pm->schedule_next_batch(8192);
    CHECK(batch == nullptr, "should reject: estimated > available");
    CHECK_EQ(f.pm->pending_size(), 1u, "still pending");
    CHECK(f.live_reqs.empty(), "no req admitted");
    f.cm.check_integrity();
}

void test_prefill_uses_prefix_match() {
    Fixture f;
    // 先分配 4 个 page,再 insert_prefix —— 模拟"另一个 req 完成后留下 prefix"。
    // 直接用 fake indices 会破坏 free + cache == num_pages 不变量。
    auto allocated = f.cm.allocate(4);
    f.cm.insert_prefix({1, 2, 3, 4}, allocated);

    // 再来一个 prompt 包含这个 prefix
    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3, 4, 5, 6}));
    auto batch = f.pm->schedule_next_batch(8192);
    CHECK(batch != nullptr, "should admit");
    auto& r = batch->reqs[0];
    CHECK_EQ(r->cached_len, 4u, "should match 4-token prefix");
    CHECK_EQ(r->device_len, 6u, "device_len is full input");
    // page_indices 的 cached 段应该被 fill 成 matched indices
    CHECK_EQ(r->page_indices[0], allocated[0], "cached page_indices[0]");
    CHECK_EQ(r->page_indices[3], allocated[3], "cached page_indices[3]");
    CHECK_EQ(r->page_indices[4], -1, "uncached page_indices[4] is placeholder");
    f.cm.check_integrity();
}

void test_prefill_abort_from_pending() {
    Fixture f;
    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3}));
    auto r = f.pm->abort_req(1);
    CHECK(!r, "abort from pending returns null (no resources held)");
    CHECK_EQ(f.pm->pending_size(), 0u, "pending drained by abort");
}

void test_prefill_drop_oversized_input() {
    Fixture f;
    // MAX_SEQ_LEN=32, input 33 token:应该直接丢,pending 不增。
    std::vector<int32_t> big(Fixture::MAX_SEQ_LEN + 1, 7);
    f.pm->add_one_req(f.make_user_msg(1, std::move(big), /*max_tokens=*/4));
    CHECK_EQ(f.pm->pending_size(), 0u, "oversized input dropped");
    CHECK(!f.pm->runnable(), "not runnable after drop");
}

void test_prefill_clamp_max_tokens() {
    Fixture f;
    // input 4 token,max_tokens=100,MAX_SEQ_LEN=32 -> 应 clamp 到 28。
    f.pm->add_one_req(f.make_user_msg(1, {1, 2, 3, 4}, /*max_tokens=*/100));
    CHECK_EQ(f.pm->pending_size(), 1u, "still admitted (just clamped)");
    auto batch = f.pm->schedule_next_batch(8192);
    CHECK(batch != nullptr, "should produce a batch");
    auto& r = batch->reqs[0];
    const size_t expected_output = Fixture::MAX_SEQ_LEN - 4;
    CHECK_EQ(r->max_device_len, 4u + expected_output, "max_tokens clamped to max_seq_len - input_len");
    f.cm.check_integrity();
}

// ============================================================================
// DecodeManager
// ============================================================================

std::shared_ptr<Req> make_decode_req(int64_t uid, size_t input_len, size_t output_len) {
    auto r = std::make_shared<Req>();
    r->uid            = uid;
    r->cached_len     = input_len;
    r->device_len     = input_len;
    r->max_device_len = input_len + output_len;
    r->input_ids.assign(input_len, 0);
    return r;
}

void test_decode_filter_and_inflight() {
    DecodeManager dm(/*page_size=*/1);
    CHECK(!dm.runnable(), "empty -> not runnable");

    auto r1 = make_decode_req(1, 5, 4);
    auto r2 = make_decode_req(2, 3, 4);
    dm.filter_reqs({r1, r2});
    CHECK(dm.runnable(), "two reqs running");
    CHECK_EQ(dm.size(), 2u, "size 2");

    // inflight_tokens = sum(remain_len) + (page_size-1)*size = (4+4) + 0 = 8
    CHECK_EQ(dm.inflight_tokens(), 8u, "inflight_tokens basic");

    // 让 r1 finished:device_len 推到 max_device_len
    r1->device_len = r1->max_device_len;
    CHECK(!r1->can_decode(), "r1 finished");

    // filter_reqs({}) 应该把 r1 从 running 中剔除
    dm.filter_reqs({});
    CHECK_EQ(dm.size(), 1u, "r1 filtered out");
}

void test_decode_schedule_batch_sorted_by_uid() {
    DecodeManager dm(1);
    auto r3 = make_decode_req(3, 5, 4);
    auto r1 = make_decode_req(1, 5, 4);
    auto r2 = make_decode_req(2, 5, 4);
    dm.filter_reqs({r3, r1, r2});

    auto b = dm.schedule_next_batch();
    CHECK(b != nullptr, "should produce decode batch");
    CHECK(b->is_decode(), "phase is decode");
    CHECK_EQ(b->reqs.size(), 3u, "three reqs");
    CHECK_EQ(b->reqs[0]->uid, 1, "sorted by uid");
    CHECK_EQ(b->reqs[1]->uid, 2, "sorted by uid");
    CHECK_EQ(b->reqs[2]->uid, 3, "sorted by uid");
}

void test_decode_abort() {
    DecodeManager dm(1);
    auto r1 = make_decode_req(1, 5, 4);
    dm.filter_reqs({r1});
    auto got = dm.abort_req(1);
    CHECK(got == r1, "abort returns the req");
    CHECK(!dm.runnable(), "running empty after abort");

    auto miss = dm.abort_req(99);
    CHECK(!miss, "abort unknown returns null");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        test_prefill_admit_one_req,
        test_prefill_token_budget_limit,
        test_prefill_cache_available_size_limit,
        test_prefill_uses_prefix_match,
        test_prefill_abort_from_pending,
        test_prefill_drop_oversized_input,
        test_prefill_clamp_max_tokens,
        test_decode_filter_and_inflight,
        test_decode_schedule_batch_sorted_by_uid,
        test_decode_abort,
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
