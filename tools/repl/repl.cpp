#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <tensortype_inc.hpp>

const size_t MEM_CTX_SIZE = 16 * 1024 * 1024 * 1024l;

bool readline(const std::string& prop, std::string& code) {
    std::cout << prop << std::flush;
    if ( std::getline(std::cin, code) ) {
        return true;
    }
    return false;
}


int main(int argc, char* argv[] ) {
    std::string text;
    for (int i = 1; i < argc; i++) {
        auto code = vt::fileToString( argv[i] );
        text = text + code + "\n";
    }

    vt::CollectiveContext::boot_pipe(0);
    vt::MemoryContext::boot( MEM_CTX_SIZE );

#ifdef _USING_DEVICE_DNNL_
    vt::ComputingContext::boot_dnnl( 0 );
#endif

#ifdef _USING_DEVICE_CUDA_
    vt::ComputingContext::boot_cuda( 0 );
#endif
#ifdef _USING_DEVICE_DCU_
    vt::ComputingContext::boot_dcu( 0 );
#endif

    vt::Enviroment* env_ = new vt::Enviroment();

    env_->execute(text);
    std::string code;
    while( readline(">> ", code ) ) {
        env_->execute( code );
    }
    delete env_;

    vt::ComputingContext::shutdown();
    vt::MemoryContext::shutdown();
    vt::CollectiveContext::shutdown();
}

