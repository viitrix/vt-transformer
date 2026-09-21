#ifndef _VT_CACHE_HPP_
#define _VT_CACHE_HPP_

// KV 缓存管理：支持 Radix/Page 等多种管理模式。
// 注意在新一代的推理引擎中, 随着序列长度的增加，KV Cache 会自动压缩/稀疏化。
// 定义 KV 缓存的结构体和相关操作，必须支持以上特征。

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace vt {

// 一个请求对应的 KV 缓存 IDX 以及 Token 输入输出
template <typename Token = int32_t, typename Index = int32_t>
struct KVSlot {
public:
    KVSlot(): idx_slot_(nullptr), _idx_slot_(nullptr),
              tok_slot_(nullptr), _tok_slot_(nullptr), offset_(-1) {
    }
    KVSlot(int offset): offset_(offset) {
    }
    
    // KV 缓存 IDX 槽位，-1 表示该位置还没有分配，CPU/CUDA(或者其他设备)各一份 
    // "_" 开始的变量，在设备端(如 CUDA)进行分配
    Index* idx_slot_;
    Index* _idx_slot_;
    
    // Token 结果跟踪槽位
    Token* tok_slot_;
    Token* _tok_slot_;

    // 同步于控制用状态标记
    int32_t *slot_flags_;
    int32_t *_slot_flags_;

    // 该 slot 在总表中的偏移（槽位序号 * kSlotSize），由 KVSlotTable 分配
    int offset() const {
        return offset_;
    }

private:
    // 根据 offset_ 从总表中分配和释放; -1 表示未挂接到任何槽位。
    // 不声明为 const, 保证 KVSlot 可默认构造/可赋值 (BatchEntry 等表元素需要)
    int offset_ = -1;
};

// SLOT 表：把 kMaxReq 个定长槽位（每个 kMaxSeqLen 个 Index）线性铺开.
// 全局的  KV 缓存 IDX 以及 Token 输入输出
template <std::size_t kMaxSeqLen, int kMaxReq, typename Token = int32_t, typename Index = int32_t>
struct KVSlotTable {
    static constexpr std::size_t kFlagsSize = 16;
    static constexpr std::size_t kSlotSize  = kMaxSeqLen;
    static constexpr std::size_t kSlotNumber = (std::size_t)kMaxReq;

    KVSlotTable(const KVSlotTable&)            = delete;
    KVSlotTable& operator=(const KVSlotTable&) = delete;

    // CPU 端与设备端的表均由外部统一分配后传入，本类不再持有/释放内存
    // flags 表同样由外部分配：kMaxReq 组，每组 kFlagsSize 个 int32_t
    KVSlotTable(Token* tok_host, Index* idx_host,
                Token* tok_device, Index* idx_device,
                int32_t* flags_host, int32_t* flags_device) {
        // 全局表：kMaxReq 个槽位，每个槽位 kMaxSeqLen 个元素，线性铺开
        idx_table_ = idx_host;
        tok_table_ = tok_host;
        _idx_table_ = idx_device;
        _tok_table_ = tok_device;
        flags_table_ = flags_host;
        _flags_table_ = flags_device;

        // 辅助变量：记录每个槽位的占用状态，请求不多，允许 O(N) 扫描
        slot_in_use_.assign(kSlotNumber, false);
        free_slot_number_ = kSlotNumber;
    }
    ~KVSlotTable() = default;

    // 分配一个空闲槽位，返回指向该槽位区间的 KVSlot；无空闲槽位时报错，按 O(N) 寻找空闲
    KVSlot<Token, Index> new_slot(){
        if (free_slot_number_ == 0) {
            vt_panic("KVSlotTable::new_slot: out of free slots!");
        }
        // O(N) 线性扫描找第一个空闲槽位
        for (std::size_t i = 0; i < kSlotNumber; i++) {
            if (!slot_in_use_[i]) {
                slot_in_use_[i] = true;
                free_slot_number_--;
                int offset = (int)(i * kSlotSize);
                KVSlot slot(offset);
                slot.idx_slot_  = idx_table_  + offset;
                slot._idx_slot_ = _idx_table_ ? _idx_table_ + offset : nullptr;
                slot.tok_slot_  = tok_table_  + offset;
                slot._tok_slot_ = _tok_table_ ? _tok_table_ + offset : nullptr;
                int flag_off = (int)(i * kFlagsSize);
                slot.slot_flags_  = flags_table_  ? flags_table_  + flag_off : nullptr;
                slot._slot_flags_ = _flags_table_ ? _flags_table_ + flag_off : nullptr;
                return slot;
            }
        }
        vt_panic("KVSlotTable::new_slot: inconsistent free slot count!");
    }

    // 释放一个已分配的槽位
    void release_slot(KVSlot<Token, Index>& slot) {
        int offset = slot.offset();
        // 偏移必须按 kSlotSize 对齐，且落在总表范围内
        if (offset < 0 || (std::size_t)offset % kSlotSize != 0) {
            vt_panic("KVSlotTable::release_slot: invalid slot offset!");
        }
        std::size_t sloti = (std::size_t)offset / kSlotSize;
        if (sloti >= kSlotNumber || !slot_in_use_[sloti]) {
            vt_panic("KVSlotTable::release_slot: double-released or unknown slot!");
        }

        slot_in_use_[sloti] = false;
        free_slot_number_++;
    }
    
private:
    // 全局表（host 端 + 设备端），以及一个激活表用于传递到设备侧
    Index* idx_table_;
    Index* _idx_table_;
    Token* tok_table_;
    Token* _tok_table_;

    // 状态标记表：kMaxReq 组，每组 kFlagsSize 个 int32_t（host 端 + 设备端）
    int32_t* flags_table_;
    int32_t* _flags_table_;

    // 根据 kSlotSize 来划分空闲和占用，辅助实现分配
    std::vector<bool> slot_in_use_;   // 槽位占用标记
    std::size_t free_slot_number_;    // 空闲槽位计数，避免整表扫尽
};


template <std::size_t kSize, int kPageSize = 4,typename Token = int32_t, typename Index = int32_t>
struct KVCachePage {
public:
    static constexpr std::size_t kCachePageNumber = kSize / kPageSize;
    static constexpr int kCachePageSize = kPageSize;
    
    KVCachePage(const KVCachePage&)            = delete;
    KVCachePage& operator=(const KVCachePage&) = delete;
    explicit KVCachePage() {
        pages_.reserve( kCachePageNumber);
        free_pages_.reserve( kCachePageNumber);

        for(std::size_t i = 0; i < kCachePageNumber; i++) {
            pages_.push_back( (Index)(i*kPageSize) );
        }
        offset_to_page_.assign(kSize, kInvalidPos);
        for(std::size_t i = 0; i < kCachePageNumber; i++) {
            offset_to_page_[ (std::size_t)(i*kPageSize) ] = i;
        }
        free_pos_.assign(kCachePageNumber, kInvalidPos);
        for(std::size_t i = 0; i < kCachePageNumber; i++) {
            free_pos_[i] = i;
            free_pages_.push_back(i);
        }
    }
    ~KVCachePage() = default;

    // 返回数据范围 [begin, end)，即该页在 pool_ 中的起始与结束偏移，不是数组。
    std::pair<Index, Index> new_page() {
        if (free_pages_.empty()) {
            vt_panic("KVCachePage::new_page: out of free pages!");
        }
        size_t pagei = free_pages_.back();
        free_pages_.pop_back();
        free_pos_[pagei] = kInvalidPos;
        Index begin = pages_[pagei];
        return { begin, begin + kPageSize };
    }

    // O(1) 释放：通过反向 map 找到页索引，通过 free_pos_ 直接判断状态并登记。
    void release_page(Index page) {
        if (page < 0 || (std::size_t)page >= kSize ) {
            vt_panic("KVCachePage::release_page: invalid page offset!");
        }

        // 不对齐的释放，不做任何处理，但需要检查对应的 offset 是否可以释放
        if ( page % kPageSize != 0 ) {
            // 该 offset 落在某一页内部：若所在页当前并未分配，
            // 则这次释放指向了空闲页内部，属于调用错误。
            size_t pagei = (std::size_t)page / kPageSize;
            if ( pagei >= free_pos_.size() ) {
                vt_panic("KVCachePage::release_page: offset falls inside no page!");
            }
            if ( free_pos_[pagei] != kInvalidPos ) {
                vt_panic("KVCachePage::release_page: offset falls inside a free page!");
            }
            return;
        }

        size_t pagei = offset_to_page_[(std::size_t)page];
        if (pagei == kInvalidPos || free_pos_[pagei] != kInvalidPos) {
            vt_panic("KVCachePage::release_page: invalid or double-released page!");
        }
        free_pos_[pagei] = free_pages_.size();
        free_pages_.push_back(pagei);
    }

private:
    static constexpr size_t kInvalidPos = static_cast<size_t>(-1);

    std::vector<Index>  pages_;
    std::vector<size_t> free_pages_;
    std::vector<size_t> offset_to_page_;   // 反向 map: pool_ 偏移 -> 页索引
    std::vector<size_t> free_pos_;         // 页索引 -> free_pages_ 中的下标（-1 为已分配）
};



} // namespace vt
#endif