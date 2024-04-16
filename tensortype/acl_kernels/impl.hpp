#ifndef _ACL_KERNELS_HPP_
#define _ACL_KERNELS_HPP_

#include <algorithm>
#include <queue>

namespace vt { namespace acl_kernels {

using device_fp16_t = __fp16;

template<typename T>
void fill_causal_mask(int* m, T* o, T minv, int full_tokens, int nt_end) {
    for ( int i = 0; i < full_tokens; i++) {
        o[i] = minv;
    }

    for ( int i = 0; i <= nt_end; i++) {
        if ( m[i] != 0 ) {
            o[i] = 0;
        }
    }
}

template <DataType DT>
void rmsnorm(ACLTensor<DT>* x, ACLTensor<DT>* scale, ACLTensor<DT>* norm2, ACLTensor<DT>* y, size_t batch_size, size_t hidden_dim, float eps) {
    if ( DT != DataType::Float &&  DT != DataType::FP16) {
        vt_panic("DNNL rmsnor only support float and fp16!");
    }
    
#pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++) {
        float rms = 0.0;    
        if ( DT == DataType::Float) {    
            for(size_t j = 0; j < hidden_dim; j++) {                
                float v = ((float *)x->data())[i * hidden_dim + j];
                rms = rms + v * v;  
            } 
        } 
        if ( DT == DataType::FP16) {    
            for(size_t j = 0; j < hidden_dim; j++) {
                float v = float(((device_fp16_t *)x->data())[i * hidden_dim + j]);
                rms = rms + v * v;
            } 
        }
  
        rms = rms / (float)hidden_dim;
        rms = 1.0 / sqrt(rms + eps);

        if ( DT == DataType::Float) {    
            for(size_t j = 0; j < hidden_dim; j++) {
                float v = ((float *)x->data())[i * hidden_dim + j];
                ((float *)y->data())[i * hidden_dim + j] = v * rms * ( ((float *)scale->data())[j]);
            } 
        } 
        if ( DT == DataType::FP16) {    
            for(size_t j = 0; j < hidden_dim; j++) {
                float v = float(((device_fp16_t *)x->data())[i * hidden_dim + j]);
                ((device_fp16_t *)y->data())[i * hidden_dim + j] = v * rms * float( ((device_fp16_t *)scale->data())[j]);
            }
        }
    }
}

template <typename T>
void rotary_embed(T* in, float* cos_sin, int* pos, T* out, size_t batch, size_t  heads, size_t tokens, size_t dims) {
    for (size_t b = 0; b < batch; b++) {
        int p = pos[b];
        #pragma omp parallel for
        for (size_t t = 0; t < tokens; t++) {
            float* tab = cos_sin + (t + p) * dims * 2;
            for (size_t h = 0; h < heads; h++) {
                size_t offset = b * heads * tokens * dims + t * heads * dims + h * dims;
                for (size_t i = 0;  i < dims/2; i++) {
                    int ii = i + dims/2;
                    float x = in[i+offset];
                    float y = in[ii+offset];
                    out[i+offset] = (tab[i*2] * x - tab[i*2+1] * y);
                    out[ii+offset] = (tab[ii*2] * y + tab[ii*2+1] * x);
                }
            }
        }
    }
}

template <typename T>
void silu_product(T* a, T* b,  T* out, size_t items);

template <>
void silu_product<float>(float* in_act, float* in,  float* out, size_t items) {
    #pragma omp parallel for
    for ( size_t i = 0; i < items; i++) {
        float act = in_act[i];
        float in_ = in[i];
        out[i] = act / (1.f + expf(-act)) * in_;
    }
}

template <>
void silu_product<device_fp16_t>(device_fp16_t* in_act, device_fp16_t* in,  device_fp16_t* out, size_t items) {
    #pragma omp parallel for
    for ( size_t i = 0; i < items; i++) {
        float act = float( in_act[i] );
        float in_ = float( in[i] );
        out[i] = act / (1.f + expf(-act)) * in_;
    }
}

template <typename T>
void easy_top1(T* logits, int* out, size_t batch, size_t vocab_size);

template <>
void easy_top1<float>(float* logits, int* out, size_t batch, size_t vocab_size) {
    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        float* src = logits + b * vocab_size;

        float max_v = std::numeric_limits<float>::min();
        int max_i = -1;
        for (int i = 0; i < (int)vocab_size; i++) {
            if ( src[i] > max_v ) {
                max_v = src[i];
                max_i = i;
            }
        }
        out[b] = max_i;
    }
}

template <>
void easy_top1<device_fp16_t>(device_fp16_t* logits, int* out, size_t batch, size_t vocab_size) {
    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        device_fp16_t* src = logits + b * vocab_size;

        float max_v = std::numeric_limits<float>::min();
        int max_i = -1;
        for (int i = 0; i < (int)vocab_size; i++) {
            float v = float( src[i] );
            if ( v > max_v ) {
                max_v = v;
                max_i = i;
            }
        }
        out[b] = max_i;
    }
}



struct TopItem {
    float v;
    int i;
    TopItem(int i_, float v_) : v(v_), i(i_) {};       
};

class Compare {
public:
    bool operator() (TopItem foo, TopItem bar) {
        return foo.v > bar.v;
    }
};


int do_sampling(std::priority_queue<TopItem, std::vector<TopItem>, Compare>& topk_, float temp, float randx) {
    std::vector<TopItem> topk;
    while ( topk_.size() > 0 ) {
        topk.push_back( topk_.top() );
        topk_.pop();
    }
    const int K = topk.size();

    float sum = 1.0;
    for(auto i = 0; i < K - 1; i++) {
        topk[i].v = expf( (topk[i].v - topk[K-1].v) / temp );
        sum = sum + topk[i].v;
    }
    topk[K-1].v = 0;
    
    float thres = 0.0;
    for(auto i = 0; i < K; i++) {
        thres += topk[i].v / sum;
        if ( thres >= randx ) {
            return topk[i].i;
        }
    }
    return topk[K-1].i;
}

template <typename T>
void easy_top3(T* logits, int* out, size_t batch, size_t vocab_size, float temp, float randx);

template <>
void easy_top3<float>(float* logits, int* out, size_t batch, size_t vocab_size, float temp, float randx) {

    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        float* src = logits + b * vocab_size;
        
        std::priority_queue<TopItem, std::vector<TopItem>, Compare> topk;
        for (int i = 0; i < 3; i++) {
            topk.push( {i, src[i]} );
        }

        for (int i = 3; i < (int)vocab_size; i++) {
            float v = src[i];
            if ( v >  topk.top().v ) {
                topk.pop();
                topk.push({i, v});
            }
        }

        out[b] = do_sampling(topk, temp, randx);
    }
    
}

template <>
void easy_top3<device_fp16_t>(device_fp16_t* logits, int* out, size_t batch, size_t vocab_size, float temp, float randx) {

    #pragma omp parallel for
    for (size_t b = 0; b < batch; b++) {
        device_fp16_t* src = logits + b * vocab_size;
        
        std::priority_queue<TopItem, std::vector<TopItem>, Compare> topk;
        for (int i = 0; i < 3; i++) {
            topk.push( {i, float(src[i])} );
        }

        for (int i = 3; i < (int)vocab_size; i++) {
            float v = float(src[i]);
            if ( v >  topk.top().v ) {
                topk.pop();
                topk.push({i, v});
            }
        }

        out[b] = do_sampling(topk, temp, randx);
    }
}



}}
#endif
