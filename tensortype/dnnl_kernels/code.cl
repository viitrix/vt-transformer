R"=====(

#pragma OPENCL EXTENSION cl_intel_printf : enable
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

)====="