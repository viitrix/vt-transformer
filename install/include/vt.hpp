#ifndef _VT_HPP_
#define _VT_HPP_

/*
 * some common interfaces used by all compoents
 */

#include <fstream>
#include <iostream>
#include <vector>

#define vt_assert(Expr, Msg) \
    vt::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define vt_panic(Msg) \
    vt::_M_Panic(__FILE__, __LINE__, Msg)

namespace vt {

const unsigned int VERSION = 0x000100;    // v0.1.0

// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "**Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "**Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
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
inline void read_data(const char* fileName, std::vector<T>& dout) {                                                                            std::ifstream inf(fileName, std::ios::binary);
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



}

#endif
