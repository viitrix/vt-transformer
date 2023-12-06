#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vt { namespace dcu {
/**
    @brief: ker_layer_norm
    Standard layer normalization.
    It will not only output the layer norm result,
    but also outputs variance.
    may also output means, depends on whether
    the means argument is nullptr

    @thread
    gridDim.x = batch_size * seq_len
    blockDim.x = hidden_size

    @param
    ln_res: [batch_size* seq_len, hidden_size], ln result.
    vars: [batch_size* seq_len], variance per token
    means: [batch_size* seq_len], means per token, can be nullput
    inp: [batch_size * seq_len, hidden_size], ln input.
    scale: [hidden_size], ln scale
    bias: [hidden_size], ln bias
*/

__global__ void ker_layer_norm_float(float *ln_res, float *vars, float *means, const float *inp,
                                    const float *scale, const float *bias, float eps ,int hidden_size) {
    // step 0. compute local sum
    float l_sum = 0;
    float l_square_sum = 0;
    const float4 *inp_f4 =
        reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;

    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float4 val = inp_f4[idx];
        l_sum += val.x + val.y + val.z + val.w;
        l_square_sum +=
            val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // step 1. compute reduce sum
    float mean_dim = float(hidden_size) * 4.f;
    float reduce_val[2] = {l_sum, l_square_sum};
    //blockReduce<ReduceType::kSum, 2>(reduce_val);

    __shared__ float s_mean, s_var;
    if (threadIdx.x == 0) {
        s_mean = reduce_val[0] / mean_dim;
        if (means != nullptr) {
            means[blockIdx.x] = s_mean;
        }
        s_var = reduce_val[1] / mean_dim - s_mean * s_mean + eps;
        vars[blockIdx.x] = s_var;
        s_var = rsqrtf(s_var);
    }
    __syncthreads();

    // step 2. layer norm result
    float4 *output_f4 =
        reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float4 vscale = __ldg(reinterpret_cast<const float4 *>(scale) + idx);
        float4 vbias = __ldg(reinterpret_cast<const float4 *>(bias) + idx);
        float4 val = inp_f4[idx];
        val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
        val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
        val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
        val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
        output_f4[idx] = val;
    }
}

}}
