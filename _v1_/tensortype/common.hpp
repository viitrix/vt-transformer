// internal shared common code
// don't export to extern
#ifndef _VT_COMMON_HPP_
#define _VT_COMMON_HPP_

#include <math.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "vt.hpp"

namespace vt {

using local_fp16_t = unsigned short;
using local_bf16_t = unsigned short;

#define SIMPLE_DUMP(d)                      \
    for(size_t i = 0; i < 8; i++) {         \
        std::cout << d[i] << " ";           \
    }                                       \
    std::cout << std::endl

#define SIMPLE_DUMP_WITH(d, convt)          \
    for(size_t i = 0; i < 8; i++) {         \
        std::cout << convt(d[i]) << " ";    \
    }                                       \
    std::cout << std::endl



float fp16_to_fp32(local_fp16_t value);
local_fp16_t fp32_to_fp16(float value);
inline float bf16_to_fp32(local_bf16_t v) {
    unsigned int proc = v << 16;
    return *reinterpret_cast<float *>(&proc);
}

inline local_bf16_t fp32_to_bf16(float val) {
    local_bf16_t d = (*reinterpret_cast<unsigned int *>(&val))>>16;
    return d;
}

inline bool
check_aligned(const void * ptr, int alignment) noexcept {
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    return !(iptr % alignment);
}

inline std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    if ( ! t.is_open() ) {
        std::cout << "Can't open " << filename << std::endl;
        vt_panic("Can't open file");
    }

    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

template<typename T>
inline void read_data(const char* fileName, std::vector<T>& dout) {
    std::ifstream inf(fileName, std::ios::binary);
    if ( ! inf.is_open() ) {
        std::cout << "Can't open " << fileName << std::endl;
        vt_panic("Can't open file");
    }

    inf.seekg(0, inf.end);
    size_t length = inf.tellg();
    inf.seekg(0, inf.beg);

    const size_t items = length / sizeof(T);
    vt_assert( items * sizeof(T) == length, "file must be aligened with item");

    dout.resize( items );
    inf.read((char *)dout.data() , sizeof(T) * items);

    inf.close();
}

inline std::string writeToFile(const char* filename, const char* data, size_t len) {
    std::ifstream t(filename);
    if ( ! t.is_open() ) {
        std::cout << "Can't open " << filename << std::endl;
        vt_panic("Can't open file");
    }

    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

template<typename T>
void fill_rotary_cache(std::vector<T>&data, int len, int dims, float base);

} // namespace vt
#endif

