#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "quantize.hpp"

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

namespace vt { namespace cuda {
__device__ float dot(float4& a, float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <const int BM, const int BN, const int THREADS, const int DIV, typename T, typename TT>
__global__ void linear2d_q4_kernel(int M, int N, int K, const T *A, const q4_block_t *B, T *C) {
    const int Abi = blockIdx.x;
    const int Bbi = blockIdx.y;
    const int KK = K / Q4_BLOCK_SIZE;

    // Move blocktile to beginning of A's row and B's column
    A += Abi * BM * K;
    B += Bbi * BN * KK;  
    C += Abi * BM * N + Bbi * BN;

    const int EM = (M - Abi * BM) >= BM ? BM : (M - Abi * BM);
    const int EN = (N - Bbi * BN) >= BN ? BN : (N - Bbi * BN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * Q4_BLOCK_SIZE ];
    __shared__ float Bs[BN * Q4_BLOCK_SIZE ];

    // used for copy data
    const int Tc = threadIdx.x % (Q4_BLOCK_SIZE/2);
    const int Tr = threadIdx.x / (Q4_BLOCK_SIZE/2);
    const int stride = THREADS / (Q4_BLOCK_SIZE/2);

    // used for computing 
    const int TM = BM / DIV;
    const int TN = BN / DIV;
    const int TNi = threadIdx.x % DIV;
    const int TMi = threadIdx.x / DIV;
    const int offsetA = TMi * TM * Q4_BLOCK_SIZE;
    const int offsetB = TNi * TN * Q4_BLOCK_SIZE;
    float tmp[TM][TN] = {0.0};
    
    for(int left = 0; left < K; left += Q4_BLOCK_SIZE) {
        // copy data from A
        TT vi;
        for (int i = Tr; i < EM; i += stride) {
            //As[i*Q4_BLOCK_SIZE + Tc] = A[i * K + Tc];
            vi = *(TT *)(&A[i * K + Tc*2]); 
            As[i*Q4_BLOCK_SIZE + Tc*2 + 0] = vi.x;
            As[i*Q4_BLOCK_SIZE + Tc*2 + 1] = vi.y;
        }
       
        // copy data from B
        const q4_block_t* q4;
        float d;
        for ( int i = Tr; i < EN; i += stride) {
            q4 = &B[i * KK];
            d = q4->d;
            
            const uint8_t vui = q4->q[Tc];
            const int8_t vi0 = vui & 0xF;
            const int8_t vi1 = vui >> 4;

            Bs[i*Q4_BLOCK_SIZE + Tc*2 + 0] = (vi0 - 8) * d;
            Bs[i*Q4_BLOCK_SIZE + Tc*2 + 1] = (vi1 - 8) * d;
        }
        __syncthreads(); 
      
        A += Q4_BLOCK_SIZE;
        B ++;

        float4 a, b;
        for (int i = 0; i < Q4_BLOCK_SIZE / 4; i++) {
            for(int m = 0; m < TM; m++) {
                a =  *(float4 *)(&As[offsetA + m * Q4_BLOCK_SIZE + i*4]);
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    //tmp[m][n] += a * Bs[ offsetB + n * Q4_BLOCK_SIZE + i]; 
                    b = *(float4 *)(&Bs[offsetB + n * Q4_BLOCK_SIZE + i*4]);  
                    tmp[m][n] += dot(a, b); 
                }
            }
        }
        __syncthreads(); 
    }
   
    const int ETM = min(EM, TMi * TM + TM);
    const int ETN = min(EN, TNi * TN + TN);

    for(int m = TMi*TM; m < ETM; m++) {
        #pragma unroll
        for (int n = TNi*TN; n < ETN; n++) {
            C[m*N + n] = tmp[m - TMi * TM][n - TNi * TN];
        }
    }
}

template <typename T>
int linear2d_q4(const T *in, const void* w, T *out, int M, int N, int K, cudaStream_t stream);

template <>
int linear2d_q4<__half>(const __half *in, const void* w, __half *out, int M, int N, int K, cudaStream_t stream) {
    const int BM = 128;
    const int BN = 64;

    const int blockThreads = 256;
    const int blockDiv = 16; 
    assert( (K % Q4_BLOCK_SIZE) == 0);

    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    linear2d_q4_kernel<BM, BN, blockThreads, blockDiv, __half, __half2> <<<grid, blockThreads, 0, stream>>> (M, N, K, in, (q4_block_t*)w, out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch linear2d_q4_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int linear2d_q4<float>(const float *in, const void* w, float *out, int M, int N, int K, cudaStream_t stream) {
    const int BM = 64;
    const int BN = 64;

    const int blockThreads = 256;
    const int blockDiv = 16; 
    assert( (K % Q4_BLOCK_SIZE) == 0);

    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    linear2d_q4_kernel<BM, BN, blockThreads, blockDiv, float, float2> <<<grid, blockThreads, 0, stream>>> (M, N, K, in, (q4_block_t*)w, out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch linear2d_q4_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <const int BM, const int BN, const int THREADS, const int DIV, typename T, typename TT>
__global__ void linear2d_q8_kernel(int M, int N, int K, const T *A, const uint8_t *B, T *C) {
    const int Abi = blockIdx.x;
    const int Bbi = blockIdx.y;
    const int KK = K + sizeof(float) * 2;

    // Move blocktile to beginning of A's row and B's column
    A += Abi * BM * K;
    B += Bbi * BN * KK;  
    C += Abi * BM * N + Bbi * BN;

    const int EM = (M - Abi * BM) >= BM ? BM : (M - Abi * BM);
    const int EN = (N - Bbi * BN) >= BN ? BN : (N - Bbi * BN);

    // allocate space for the current blocktile in smem
    const int MOVING_BLOCK = 32;
    __shared__ float  As[BM * MOVING_BLOCK];
    __shared__ float  Bs[BN * MOVING_BLOCK];

    // used for copy data
    const int Tc = threadIdx.x % (MOVING_BLOCK/2);
    const int Tr = threadIdx.x / (MOVING_BLOCK/2);
    const int stride = THREADS / (MOVING_BLOCK/2);

    // used for computing 
    const int TM = BM / DIV;
    const int TN = BN / DIV;
    const int TNi = threadIdx.x % DIV;
    const int TMi = threadIdx.x / DIV;
    const int offsetA = TMi * TM * MOVING_BLOCK;
    const int offsetB = TNi * TN * MOVING_BLOCK;
    float tmp[TM][TN] = {0.0};
 

    for(int left = 0; left < K; left += MOVING_BLOCK) {
        // copy data from A
        TT vi;
        for (int i = Tr; i < EM; i += stride) {
            vi = *(TT *)(&A[i * K + Tc*2]); 
            As[i*MOVING_BLOCK + Tc*2 + 0] = vi.x;
            As[i*MOVING_BLOCK + Tc*2 + 1] = vi.y;
        }
       
        // copy data from B
        for ( int i = Tr; i < EN; i += stride) {
            q8_head_t* q8 = (q8_head_t *)(B + i * KK);   
            
            unsigned short bb = *(unsigned short*)(&q8->q[left + Tc*2]);
            uint8_t low = bb & 0xFF;
            uint8_t high = bb >> 8;
            Bs[i*MOVING_BLOCK + Tc*2 + 0] = low * q8->d + q8->m;    
            Bs[i*MOVING_BLOCK + Tc*2 + 1] = high * q8->d + q8->m;
        }
        __syncthreads(); 
      
        A += MOVING_BLOCK;
    
        float4 a, b;
        for (int i = 0; i < MOVING_BLOCK / 4; i++) {
            for(int m = 0; m < TM; m++) {
                a =  *(float4 *)(&As[offsetA + m * MOVING_BLOCK + i*4]);
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    b = *(float4 *)(&Bs[offsetB + n * MOVING_BLOCK + i*4]);  
                    tmp[m][n] += dot(a, b); 
                }
            }
        }
        __syncthreads(); 
    }
   
    const int ETM = min(EM, TMi * TM + TM);
    const int ETN = min(EN, TNi * TN + TN);

    for(int m = TMi*TM; m < ETM; m++) {
        #pragma unroll
        for (int n = TNi*TN; n < ETN; n++) {
            C[m*N + n] = tmp[m - TMi * TM][n - TNi * TN];
        }
    }
}

template <typename T>
int linear2d_q8(const T *in, const void* w, T *out, int M, int N, int K, cudaStream_t stream);

template <>
int linear2d_q8<__half>(const __half *in, const void* w, __half *out, int M, int N, int K, cudaStream_t stream) {
    const int BM = 128;
    const int BN = 128;

    const int blockThreads = 256;
    const int blockDiv = 16; 

    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    linear2d_q8_kernel<BM, BN, blockThreads, blockDiv, __half, __half2> <<<grid, blockThreads, 0, stream>>> (M, N, K, in, (uint8_t*)w, out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch linear2d_q8_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


template <>
int linear2d_q8<float>(const float *in, const void* w, float *out, int M, int N, int K, cudaStream_t stream) {
    const int BM = 128;
    const int BN = 128;

    const int blockThreads = 256;
    const int blockDiv = 16; 

    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    linear2d_q8_kernel<BM, BN, blockThreads, blockDiv, float, float2> <<<grid, blockThreads, 0, stream>>> (M, N, K, in, (uint8_t*)w, out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch linear2d_q8_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}} // endof cuda
