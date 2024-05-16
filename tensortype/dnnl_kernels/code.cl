R"=====(


__kernel void rmsnorm_fp16(__global const half *feature, __global  const half *w, __global  half *out, __global  half *norm2,
                               const int batch, const int dim, const float eps) {    
    const int GROUP_SIZE = 16;
    int b = get_group_id(0);
    int tid = get_local_id(0);
    
    __global const half* src = feature + b * dim;
    __global half* dst = out + b * dim;

    __local float s[GROUP_SIZE];
    __local float srms;

    float rms = 0.0;
    for(int i = tid; i < dim; i += GROUP_SIZE) {
        rms = rms + (src[i] * src[i]);
    }
    s[tid] = rms;
    barrier(CLK_LOCAL_MEM_FENCE);

    if ( tid == 0 ) {
        rms = 0.0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            rms = rms + s[i];
        }
        rms = rms / (float)dim;
        rms = 1.0 / sqrt(rms + eps);
        srms = rms;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    rms = srms;
    for(int i = tid; i < dim; i += GROUP_SIZE) {
        dst[i] = rms * src[i] * w[i];
    }
}

#define Q8_BLOCK_SIZE 1024

__kernel void linear_fp16_w8(const __global half* A, const __global uchar* B, __global const float* Bscale,  __global half* C, const __global half* bias, 
                          const int BATCH, const int OUT, const int IN, const int using_bias) {
    const int TB = 4;
    int b = get_local_id(0);    // 0..TB
    int o = get_local_id(1);
    int gb = get_group_id(0) * TB;
    int go = get_group_id(1) * TB;

    __local float SubA[TB][TB];
    __local float SubB[TB][TB];

    float acc = 0;
    float minv;
    float scale;
    for (int i = 0; i < IN; i += TB) {
        int si = ((go + b) * IN + i + o) / Q8_BLOCK_SIZE;
        scale = Bscale[si * 2 + 1];
        minv =  Bscale[si * 2 ];

        SubA[b][o] = A[ (gb + b) * IN + i + o];
        SubB[b][o] = B[ (go + b) * IN + i + o] * scale + minv;

        barrier(CLK_LOCAL_MEM_FENCE);

        acc += SubA[b][0] * SubB[o][0];
        acc += SubA[b][1] * SubB[o][1];
        acc += SubA[b][2] * SubB[o][2];
        acc += SubA[b][3] * SubB[o][3];

        barrier(CLK_LOCAL_MEM_FENCE); 
    } 
    
    int batch = get_global_id(0);
    int out = get_global_id(1);
    if( using_bias ) {
        C[ batch * OUT + out] = acc + bias[out];
    } else {
        C[ batch * OUT + out] = acc;
    }
}

__kernel void linear_fp16(const __global half* A, const __global half* B, __global half* C, const __global half* bias, 
                          const int BATCH, const int OUT, const int IN, const int using_bias) {
    const int TB = 4;
    int b = get_local_id(0);    // 0..TB
    int o = get_local_id(1);
    int gb = get_group_id(0) * TB;
    int go = get_group_id(1) * TB;

    __local half SubA[TB][TB];
    __local half SubB[TB][TB];

    float acc = 0;
    for (int i = 0; i < IN; i += TB) {
        SubA[b][o] = A[ (gb + b) * IN + i + o];
        SubB[b][o] = B[ (go + b) * IN + i + o];

        barrier(CLK_LOCAL_MEM_FENCE);

        acc += SubA[b][0] * SubB[o][0];
        acc += SubA[b][1] * SubB[o][1];
        acc += SubA[b][2] * SubB[o][2];
        acc += SubA[b][3] * SubB[o][3];

        barrier(CLK_LOCAL_MEM_FENCE); 
    } 
    
    int batch = get_global_id(0);
    int out = get_global_id(1);
    if( using_bias ) {
        C[ batch * OUT + out] = acc + bias[out];
    } else {
        C[ batch * OUT + out] = acc;
    }
}

__kernel void rotary_embed_fp16(const __global half *in, const __global float *cos_sin, const __global int* pos, __global half *out,
                                   const int bs, const int hnum, const int len, const int dims) {
    int e = get_global_id(0);
    if ( e > bs * hnum * dims) {
        return;
    }

    in = in + e * dims;
    out = out + e * dims;

    int b = e / (len * hnum);
    int l = (e - b * len * hnum) / hnum + pos[b];
    cos_sin = cos_sin + l * dims * 2;

    for (int i = 0; i < dims / 2; i++) {
        int ii = i + dims/2;
        float x = in[i];
        float y = in[ii];
        out[i] = (cos_sin[i*2] * x - cos_sin[i*2+1] * y);
        out[ii] = (cos_sin[ii*2] * y + cos_sin[ii*2+1] * x);
    }
}

__kernel void transpose_0213_fp16(const __global half *in, __global half *out,
                                   const int A, const int B, const int C, const int D) {
    
    size_t id = get_global_id(0);

    int d = id % D; id = id / D;
    int c = id % C; id = id / C;
    int b = id % B;
    int a = id / B;

    id = get_global_id(0);
    size_t to = a * B * C * D + c * B * D + b * D + d;
    out[to] = in[id];
}

__kernel void dequantize_fp16(__global const uchar *in, __global const float* scale, __global half *out,
                                   const int feature) {
    size_t id = get_global_id(0);
    size_t sid = id / feature;

    out[id] = in[id] * scale[sid * 2 + 1] + scale[sid * 2];
}

__kernel void quantize_fp16(__global const half *in, __global float* scale, __global uchar *out,
                                   const int feature) {
    size_t f = get_global_id(0);
    
    in = in + f * feature;
    out = out + f * feature;

    float maxv = 0.0;
    float minv = 0.0;
    for (int i = 0; i < feature; i++) {
        float v = in[i];
        if ( v > maxv) {
            maxv = v;
        }
        if ( v < minv ) {
            minv = v;
        }
    }

    float s = (maxv - minv) / 255.0;
    scale[f * 2] = minv;
    scale[f * 2 + 1] = s;

    const float id = (s != 0.0) ? 1.0 / s : 0.0f;
     
    for(size_t i = 0; i < feature; i++) {
        float v = (in[i] - minv)* id;
        out[i] = (uchar)(v + 0.5);
    }
}

)====="
