#ifndef _VT_COMMON_HPP_
#define _VT_COMMON_HPP_

#include <iostream>

namespace vt {
const unsigned int VERSION = 0x000020;    // v0.2.0

// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "**Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

[[noreturn]] inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "**Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}

} // end of namespace

#define vt_assert(Expr, Msg) \
    vt::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define vt_panic(Msg) \
    vt::_M_Panic(__FILE__, __LINE__, Msg)

#define vt_fatal_error() \
    vt::_M_Panic(__FILE__, __LINE__, "Fatal error, can't be here")


#endif