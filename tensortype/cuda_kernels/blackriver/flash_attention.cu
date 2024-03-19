#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

// just declare
namespace vt { namespace cuda {
template <typename T>
int flash_attention(const T* query, T* key, T* value, int batch, int heads, int length, int hidden, cudaStream_t stream);
}}

namespace vt { namespace cuda {
    
}}

