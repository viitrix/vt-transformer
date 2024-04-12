#ifndef _ACL_KERNELS_HPP_
#define _ACL_KERNELS_HPP_

#include <algorithm>
#include <queue>

namespace vt { namespace acl_kernels {

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



}}
#endif
