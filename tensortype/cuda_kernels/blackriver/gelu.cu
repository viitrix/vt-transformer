#include "kernels.h"

/*
def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g
*/

namespace vt { namespace cuda {
template <typename T>
int gelu_forward(const T* src, T* target, int nElementNumber, cudaStream_t stream);

template<typename T>
__global__ void gelu(T* target, const T* src, int nElementNumber) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nElementNumber) {
        float value = src[i];
        //target[i] = value * (0.5F + 0.5F * tanhf(value * (0.79788456F + 0.03567741F * value * value)));
        target[i] = value * normcdf(value);
    }
}


template <>
int gelu_forward<float>(const float* src, float* target, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    gelu<float> <<< num_of_blocks, block_size, 0, stream >>> (target, src, nElementNumber);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return 0;
}

template <>
int gelu_forward<__half>(const __half* src, __half* target, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    gelu<__half> <<< num_of_blocks, block_size, 0, stream >>> (target, src, nElementNumber);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return 0;
}


__global__ void gelu_bw(const float* out_g, const float* xi, float* x_g, int nElementNumber) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nElementNumber) {
        float x = xi[i];
        float tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x));
        float ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out);
        x_g[i] = ff * out_g[i];
    }
}

int gelu_backward(const float* out_g, const float* xi, float* x_g, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    gelu_bw <<< num_of_blocks, block_size, 0, stream >>> (out_g, xi, x_g, nElementNumber);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return 0;
}

}}
