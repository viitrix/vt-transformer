R"=====(


__kernel void rmsnorm_kernel(__global const half *feature, __global  const half *w, __global  half *out, __global  half *norm2,
                               const int batch, const int dim, const float eps) {    
    int b = get_global_id(0);
    if ( b >= batch) {
        return;
    }

    __global const  half* src = feature + b * dim;
    __global half* dst = out + b * dim;
    
    float rms = 0.0;
    for(int i = 0; i < dim; i++) {
        rms = rms + (src[i] * src[i]);
    }

    rms = rms / (float)dim;
    rms = 1.0 / sqrt(rms + eps);

    for(int i = 0; i < dim; i++) {
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

)====="