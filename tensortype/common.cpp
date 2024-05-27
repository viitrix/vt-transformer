
#include "common.hpp"

namespace vt {

template<>
void fill_rotary_cache<float>(std::vector<float>&data, int len, int dims,float base) {
    std::vector<float> inv_freq;
    inv_freq.resize(dims);
    for (int i = 0; i < dims ; i += 2) {
        float f = 1.0 / pow(base,  1.0 * i / dims);
        inv_freq[i / 2] = f;
        inv_freq[dims / 2 + i / 2] = f;
    }

    for (int l = 0; l < len; l++ ) {
        for (int i = 0; i < dims; i++) {
            //freqs.push_back( 1.0 * i * inv_freq[i] );
            float f = 1.0 * l * inv_freq[i];
            data.push_back( cos(f) );
            data.push_back( sin(f) );
        }
    }
}

}

