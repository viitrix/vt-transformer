#ifndef _VT_SCHEDUER_HPP_
#define _VT_SCHEDUER_HPP_

#include <algorithm>

#include "common.hpp"
#include "request.hpp"
#include "cache.hpp"
#include "radix.hpp"
#include "engine.hpp"
#include "backend.hpp"
#include "vt.hpp"

namespace vt {

template <std::size_t kMaxSeqLen, int kMaxReq, int kPageSize = 4, typename Token = int32_t, typename Index = int32_t>
struct Scheduler {
public:
    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    explicit Scheduler(Token* tok_host, Index* idx_host,
                       Token* tok_device, Index* idx_device,
                       int32_t* flags_host, int32_t* flags_device,
                       BatchTable<kMaxReq,Token, Index>* host_batch,
                       BatchTable<kMaxReq,Token, Index>* device_batch,
                       EngineBase<kMaxSeqLen,kMaxReq,kPageSize,Token,Index>* eng,
                       BackendBase<Token>* backend) {
        
        backend_ = backend;
        engine_ = eng;
        // host/device 端的 BatchTable 均由外部统一分配后传入
        batch_  = host_batch;
        _batch_ = device_batch;

        // slot 表: kMaxReq 个槽位, 每槽 kMaxSeqLen; page 池覆盖全部槽位空间
        slot_tab_ = new KVSlotTable<kMaxSeqLen, kMaxReq, Token, Index>(
            tok_host, idx_host, tok_device, idx_device,
            flags_host, flags_device);
        pages_    = new KVCachePage<kMaxSeqLen * kMaxReq, kPageSize, Token, Index>();
        radix_    = new RadixTree<kPageSize, Token, Index>();
    }

    ~Scheduler() {
        delete slot_tab_;
        delete pages_;
        delete radix_;
    }
    
    
    int main_overlap_loop();
    
private:
    BackendBase<Token>* backend_;
    EngineBase<kMaxSeqLen, kMaxReq, kPageSize, Token, Index>* engine_;
    BatchTable<kMaxReq, Token, Index>* batch_;
    BatchTable<kMaxReq, Token, Index>* _batch_;
  
    KVSlotTable<kMaxSeqLen, kMaxReq, Token, Index>* slot_tab_;
    KVCachePage<kMaxSeqLen * kMaxReq, kPageSize, Token, Index>* pages_;
    RadixTree<kPageSize, Token, Index>* radix_;
};


} // namespace vt

#endif
