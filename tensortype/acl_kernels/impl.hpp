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
                float v = fp16_to_fp32(((device_fp16_t *)x->data())[i * hidden_dim + j]);
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


}}
#endif
