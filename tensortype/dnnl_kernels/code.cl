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

#define TS  16
__kernel void linear_fp16(const __global half* A, const __global half* B, __global half* C, const __global half* bias, 
                          const int BATCH, const int OUT, const int IN, const int using_bias) {
    
    // Thread identifiers
    const int si = get_local_id(1);        // 0.. TS 
    const int batch = get_group_id(0);     // 0.. BATCH
    const int out = get_group_id(1);       // 0.. OUT

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float accs[TS];
 
    // Initialise the accumulation registers
    float accWPT = 0.0f;

    const int numTiles = IN/TS;
    for (int t = 0; t < numTiles; t++) {
        float a = A[ batch * IN + t * TS + si ];
        float b = B[ out * IN + t * TS + si];    
        accWPT += a * b; 
    }
    accs[si] = accWPT;
    barrier(CLK_LOCAL_MEM_FENCE);

    if ( si == 0 ) {
        float sum = 0.0;
        for (int i = 0; i < TS; i++) {
            sum += accs[i];
        }        
        if ( using_bias ) {
            C[batch * OUT + out] = sum + bias[out];
        } else {
            C[batch * OUT + out] = sum;
        }
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

)====="