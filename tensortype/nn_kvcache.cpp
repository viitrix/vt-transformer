#include <chrono>
#include "tensortype.hpp"
#include "context.hpp"
#include "dag.hpp"

namespace vt {

namespace nn {
    struct EasyKVCache {
        struct KVCacheEntry {
            std::vector<int> id_;
            int begin_;            // first one valid
            int seq_;              // sequence number
            int end_;              // last one valid
            int invalid_;          // first one should be filled

            int get_cached() {
                vt_assert( (begin_ >= 0) && (end_ >= 0) && (invalid_ >= 0), "Finding a invalid KVCacheEntry ");
                if ( invalid_ >= begin_ ) {
                    return invalid_ - begin_;
                }
                return (int)id_.size() + invalid_ - begin_;
            }

            int get_uncached() {
                vt_assert( (begin_ >= 0) && (end_ >= 0) && (invalid_ >= 0), "Finding a invalid KVCacheEntry ");

                if ( end_ >= invalid_ ) {
                    return end_ - invalid_ + 1;
                }
                return (int)id_.size() + end_ - invalid_ + 1;
            }

            void copy_cached(tensor_t cache_, tensor_t left_) {
                vt_assert( invalid_ != begin_ , "Can't copy zero cached");
                int hidden_size = cache_->shape()[1];

                if ( invalid_ > begin_ ) {
                    int cached_len = get_cached();
                    std::vector<size_t> blk_shape{(size_t)cached_len, (size_t)hidden_size};
                    size_t offset = begin_ * hidden_size;
                    tensor_t src = std::get<1>(cache_->op_view(cache_, offset, blk_shape));

                    left_->op_copy(left_, src);
                } else {
                    // once
                    int len1 = id_.size() - begin_;
                    std::vector<size_t> blk_shape{(size_t)len1, (size_t)hidden_size};
                    size_t offset = begin_ * hidden_size;
                    if ( blk_shape[0] > 0) {
                        tensor_t src = std::get<1>(cache_->op_view(cache_, offset, blk_shape));
                        tensor_t dst = std::get<1>(left_->op_view(left_, 0, blk_shape));
                        dst->op_copy(dst, src);
                    }

                    // twice
                    blk_shape[0] = invalid_;
                    if ( blk_shape[0] > 0) {
                        tensor_t src = std::get<1>(cache_->op_view(cache_, 0, blk_shape));
                        tensor_t dst = std::get<1>(left_->op_view(left_, dst->items(), blk_shape));
                        dst->op_copy(dst, src);
                    }
                }
            }

            void add_uncached(tensor_t cache_, tensor_t new_) {
                int hidden_size = cache_->shape()[1];
                int uncached = get_uncached();

                if ( end_ >= invalid_ ) {
                    std::vector<size_t> blk_shape{(size_t)uncached, (size_t)hidden_size};
                    tensor_t src = std::get<1>(new_->op_view(new_, 0, blk_shape));

                    size_t offset = invalid_ * hidden_size;
                    tensor_t dst = std::get<1>(cache_->op_view(cache_, offset, blk_shape));
                    dst->op_copy(dst, src);
                } else {
                    // once
                    int len1 = id_.size() - invalid_;
                    std::vector<size_t> blk_shape{(size_t)len1, (size_t)hidden_size};
                    size_t offset = invalid_ * hidden_size;
                    if ( blk_shape[0] > 0) {
                        tensor_t dst = std::get<1>(cache_->op_view(cache_, offset, blk_shape));
                        tensor_t src = std::get<1>(new_->op_view(new_, 0, blk_shape));
                        dst->op_copy(dst, src);
                    }

                    // twice
                    blk_shape[0] = uncached - len1;
                    if ( blk_shape[0] > 0 ) {
                        tensor_t src = std::get<1>(new_->op_view(new_, src->items(), blk_shape));
                        tensor_t dst = std::get<1>(cache_->op_view(cache_, 0, blk_shape));
                        dst->op_copy(dst, src);
                    }
                }
            }

            void replace(const int tokens, const int* id, const int* mask) {
                begin_ = 0;
                end_ = 0;
                invalid_ = 0;
                seq_ = 0;
                id_[0] = id[0];

                for (int i = 1; i < tokens; i++) {
                    if ( mask[i] == 0 ) {
                        break;
                    }
                    int ii = i % id_.size();
                    if ( ii == begin_ ) {
                        vt_panic("Input tokens is too long!");
                    }
                    id_[ii] = id[i];
                    end_ = ii;
                }
            }

            int append(const int tokens, const int* id, const int* mask, const int matched_len, const int matched_begin) {
                // updating seq_ to matched_begin
                for (int i = begin_;  ; i++) {
                    int ii = i % (int)id_.size();
                    if ( ii != matched_begin ) {
                        seq_++;
                    } else {
                        break;
                    }
                }

                begin_ = matched_begin;
                end_ = -1;
                invalid_ = -1;

                for (int i = 0; i < tokens; i++) {
                    if ( mask[i] == 0 ) {
                        break;
                    }
                    int ii = ( i + begin_ ) % id_.size();
                    id_[ii] = id[i];
                    end_ = ii;

                    if ( i == matched_len ) {
                        invalid_ = end_;
                    }
                }

                // at least uncache one
                if ( invalid_ == -1 ) {
                    invalid_ = end_;
                    return matched_len - 1;
                }
                return matched_len;
            }

            std::tuple<int, int> match(const int tokens, const int* id, const int* mask) {      // return {length, begin}
                if ( begin_ == -1 ) {
                    return {0, -1};
                }

                int best_length = 0;
                int best_begin = -1;

                for (int bi = begin_; ; ) {
                    int len = 0;
                    for ( int i = 0; i < tokens; i++) {
                        if ( mask[i] == 0) {
                            break;
                        }
                        int ii = bi + i;
                        ii = ii % (int)id_.size();
                        if ( id_[ii] == id[i] ) {
                            len++;
                        } else {
                            break;
                        }
                        if ( ii == end_ ) {
                            break;
                        }
                    }

                    if ( len > best_length ) {
                        best_length = len;
                        best_begin = bi;
                    }

                    // checking out of loop
                    if ( bi == end_ ) {
                        break;
                    }
                    bi = bi + 1;
                    bi = bi % (int)id_.size();
                }

                return {best_length, best_begin};
            }
        };

        const int hidden_size;
        const int cached_tokens;
        const int cached_number;
        const int cached_layer;

        std::vector<KVCacheEntry> all_caches_;
        std::vector<int> batched_caches_;
        std::list<int> ordered_caches_;
        int left_max;
        int right_max;

        EasyKVCache(int hs, int ct, int cn, int cl) : hidden_size(hs), cached_tokens(ct), cached_number(cn), cached_layer(cl) {
            for (int i = 0; i < cn; i++) {
                KVCacheEntry kvc;
                kvc.id_.resize(cached_tokens);
                kvc.begin_ = -1;
                kvc.end_ = -1;
                kvc.invalid_ = -1;

                all_caches_.push_back( kvc );
                ordered_caches_.push_back(i);
            }
        }

        void reset() {
            for (size_t i = 0; i < batched_caches_.size(); i++) {
                ordered_caches_.push_back( batched_caches_[i] );
            }
            batched_caches_.clear();
            right_max = -1;
            left_max = -1;
        }

        int do_match(const int tokens, const int* id, const int* mask) {
            std::tuple<int, int> match_result{0, -1};
            auto best_matched = ordered_caches_.end();
            for (auto ii = ordered_caches_.begin(); ii != ordered_caches_.end(); ii++) {
                int i = *ii;
                auto ret = all_caches_[i].match(tokens, id, mask);
                if ( std::get<0>(ret) > std::get<0>(match_result) ) {
                    match_result = ret;
                    best_matched = ii;
                }
            }

            if ( best_matched ==  ordered_caches_.end() ) {
                int i = ordered_caches_.front();
                ordered_caches_.pop_front();
                batched_caches_.push_back(i);
                all_caches_[i].replace(tokens, id, mask);
                return 0;
            } else {
                int i  = *best_matched;
                ordered_caches_.erase(best_matched);
                batched_caches_.push_back(i);
                int len = all_caches_[i].append(tokens, id, mask, std::get<0>(match_result), std::get<1>(match_result) );
                return len;
            }
        }

        tensor_t get_sub_cache(tensor_t kv_cache, int b, int l ) {
            int ci = batched_caches_[b];
            size_t offset = l * cached_number * cached_tokens * hidden_size + ci * cached_tokens * hidden_size;
            std::vector<size_t> sub_shape{(size_t)cached_tokens, (size_t)hidden_size};

            auto ret = kv_cache->op_view(kv_cache, offset, sub_shape);
            return std::get<1>(ret);
        }

        void do_update(int b, tensor_t full_, tensor_t new_, tensor_t cache_) {
            int ci = batched_caches_[b];
            all_caches_[ci].add_uncached(cache_, new_);

            int cached_len = all_caches_[ci].get_cached();
            //int uncached_len = all_caches_[ci].get_uncached();
            if ( cached_len > 0 ) {
                int left_begin = left_max - cached_len;
                vt_assert(left_begin >= 0, "Can't bhere!");
                std::vector<size_t> left_shape{(size_t)cached_len, (size_t)hidden_size};
                tensor_t left_ = std::get<1>(full_->op_view(full_, left_begin * hidden_size, left_shape));

                all_caches_[ci].copy_cached(cache_, left_);
            }

            {
                tensor_t dst = std::get<1>( full_->op_view(full_, left_max * hidden_size, new_->shape().vec() ) );
                dst->op_copy(dst, new_);
            }
        }
    };

    struct EasyKVCacheInit : public NativeWord {
        void run(Stack& stack) override {
            tensor_t vcache = stack.pop_tensor();
            tensor_t kcache = stack.pop_tensor();

            vt_assert(vcache->shape() == kcache->shape() , "K&V cache must have same size!");
            int cached_layer = vcache->shape()[0];
            int cached_number = vcache->shape()[1];
            int cached_tokens = vcache->shape()[2];
            int hidden_size = vcache->shape()[3];
            EasyKVCache* cache = new EasyKVCache( hidden_size, cached_tokens, cached_number, cached_layer);

            // pass object's address to tensor
            std::vector<size_t> obj_shape;
            obj_shape.push_back( sizeof(EasyKVCache *) );
            tensor_t obj_t = vt::create_host_int(obj_shape);
            memcpy((char *)obj_t->device_data(), (char *)&cache, sizeof(EasyKVCache *));
            stack.push_tensor(obj_t);
        }
        NWORD_CREATOR_DEFINE_LR(EasyKVCacheInit)
    };

    struct EasyKVCacheMatch : public NativeWord {
        void run(Stack& stack) override {
            tensor_t _mask = stack.pop_tensor();
            tensor_t _ids = stack.pop_tensor();
            tensor_t mask_ = stack.pop_tensor();
            tensor_t ids_ = stack.pop_tensor();
            tensor_t obj_t = stack.pop_tensor();

            EasyKVCache* cache_man;
            memcpy( (char *)&cache_man, (char *)obj_t->device_data(), sizeof(EasyKVCache *));
            cache_man->reset();

            const int batch = mask_->shape()[0];
            const int tokens = mask_->shape()[1];

            std::vector<int> left_cached;
            std::vector<int> right_uncached;
            for (int b = 0; b < batch; b++) {
                int* id = (int *)ids_->device_data() + b * tokens;
                int* mask = (int *)mask_->device_data() + b * tokens;

                int cached = cache_man->do_match(tokens, id, mask);
                int uncached = tokens - cached;
                for ( int i = tokens - 1; i >= 0; i--) {
                    if ( mask[i] == 0 ) {
                        uncached = uncached - 1;
                    } else {
                        break;
                    }
                }
                left_cached.push_back( cached);
                right_uncached.push_back( uncached);
            }

            // store left/right max length
            int right_max = *std::max_element(right_uncached.begin(), right_uncached.end());
            int left_max = *std::max_element(left_cached.begin(), left_cached.end());
            vt_assert( right_max > 0, "Uncached tokens can't be zero");

            // force matrix shape has even token number
            if ( right_max % 4 != 0) {
                int padding = 4 - (right_max % 4);
                right_max = right_max + padding;
            }
            cache_man->right_max = right_max;
            cache_man->left_max = left_max;

            // create layout of new tokens
            std::vector<size_t> right_shape{(size_t)batch, (size_t)right_max};
            tensor_t right_ = vt::create_host_int(right_shape);
            for (int b = 0; b < batch; b++) {
                int* id = (int *)ids_->device_data() + b * tokens;
                int* nid = (int *)right_->device_data() + b * right_max;

                int left_length = left_cached[b];
                int right_length = right_uncached[b];
                for (int i = 0; i < right_length; i++) {
                    nid[i] = id[left_length + i];
                }
                for (int i = right_length; i < right_max; i++) {
                    nid[i] = id[tokens-1];
                }
            }

            tensor_t ids = std::get<1>(_ids->op_view(_ids, 0, right_shape));
            ids->op_copy(ids, right_);

            // create layout of mask
            std::vector<size_t> left_shape{(size_t)batch, (size_t)(left_max + right_max)};
            tensor_t left_ = vt::create_host_int(left_shape);
            for (int b = 0; b < batch; b++) {
                int* m = (int *)mask_->device_data() + b * tokens;
                int* nm = (int *)left_->device_data() + b * (right_max + left_max);

                int left_length = left_cached[b];
                int right_length = right_uncached[b];
                for (int i = 0; i < left_max - left_length; i++) {
                    nm[i] = 0;
                }
                for (int i = left_max - left_length;  i < left_max + right_length; i++) {
                    int ii = i - (left_max - left_length);
                    nm[i] = m[ii];
                }
                for (int i = left_max + right_length; i < left_max + right_max;  i++) {
                    nm[i] = 0;
                }
            }

            tensor_t mask = std::get<1>(_mask->op_view(_mask, 0, left_shape));
            mask->op_copy(mask, left_);

            stack.push_tensor(ids);
            stack.push_tensor(mask);
        }

        NWORD_CREATOR_DEFINE_LR(EasyKVCacheMatch)
    };

    struct EasyKVCachePosition : public NativeWord {
        void run(Stack& stack) override {
            tensor_t _pos = stack.pop_tensor();
            tensor_t obj_t = stack.pop_tensor();

            EasyKVCache* cache_man;
            memcpy( (char *)&cache_man, (char *)obj_t->device_data(), sizeof(EasyKVCache *));

            int batch = cache_man->batched_caches_.size();
            std::vector<size_t> shape{(size_t)batch};
            tensor_t pos_ = vt::create_host_int(shape);
            tensor_t pos = std::get<1>(_pos->op_view(_pos, 0, shape));

            int* p = (int *)pos_->device_data();
            for (int b = 0; b < batch; b++) {
                int ci = cache_man->batched_caches_[b];
                int seq = cache_man->all_caches_[ci].seq_ + cache_man->all_caches_[ci].get_cached();
                p[b] = seq;
            }
            pos->op_copy(pos, pos_);

            stack.push_tensor(pos);
        }

        NWORD_CREATOR_DEFINE_LR(EasyKVCachePosition)
    };

    struct EasyKVCacheUpdate : public NativeWord {
        void run(Stack& stack) override {
            int kv_layer = stack.pop_number();
            tensor_t kv_full = stack.pop_tensor();
            tensor_t kv_new = stack.pop_tensor();
            tensor_t kv_cache = stack.pop_tensor();
            tensor_t obj_t = stack.pop_tensor();

            EasyKVCache* cache_man;
            memcpy( (char *)&cache_man, (char *)obj_t->device_data(), sizeof(EasyKVCache *));

            vt_assert( (kv_layer >= 0) && (kv_layer < cache_man->cached_layer), "It's out of size of ordered_batched_ ");
            vt_assert( (int)kv_cache->shape()[0] == cache_man->cached_layer, "kvcache must has same size with init");
            vt_assert( (int)kv_cache->shape()[1] == cache_man->cached_number, "kvcache must has same size with init");
            vt_assert( (int)kv_cache->shape()[2] == cache_man->cached_tokens, "kvcache must has same size with init");
            vt_assert( (int)kv_cache->shape()[3] == cache_man->hidden_size, "kvcache must has same size with init");

            int batches = kv_full->shape()[0];
            int full_tokens = kv_full->shape()[1];
            int new_tokens = kv_new->shape()[1];
            int hidden_size = kv_full->shape()[2];
            std::vector<size_t> full_shape{(size_t)full_tokens, (size_t)hidden_size};
            std::vector<size_t> new_shape{(size_t)new_tokens, (size_t)hidden_size};
            for (int b = 0; b < batches; b++ ) {
                tensor_t full_ = std::get<1>(kv_full->op_view(kv_full, b * full_tokens * hidden_size, full_shape));
                tensor_t new_ = std::get<1>(kv_new->op_view(kv_new, b * new_tokens * hidden_size, new_shape));
                tensor_t cache_ = cache_man->get_sub_cache(kv_cache, b, kv_layer);
                //full_->op_zero(full_);
                cache_man->do_update(b, full_, new_, cache_);
            }
        }

        NWORD_CREATOR_DEFINE_LR(EasyKVCacheUpdate)
    };
}

void load_nn_kvcache(Enviroment& env) {
    env.insert_native_word("nn.ezkv_init", nn::EasyKVCacheInit::creator);
    env.insert_native_word("nn.ezkv_match", nn::EasyKVCacheMatch::creator);
    env.insert_native_word("nn.ezkv_position", nn::EasyKVCachePosition::creator);
    env.insert_native_word("nn.ezkv_update", nn::EasyKVCacheUpdate::creator);
}

}// end of namespace br
