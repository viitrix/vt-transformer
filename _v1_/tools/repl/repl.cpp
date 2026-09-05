#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <tensortype_inc.hpp>
#include <common.hpp>

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

    vt::ComputingContext* ctx_ = new vt::ComputingContext();
    ctx_->boot_host(0);

#ifdef _USING_DEVICE_CUDA_
    ctx_->boot_cuda(0);
#endif

#ifdef _USING_DEVICE_HIP_
    ctx_->boot_hip(0);
#endif


    vt::Enviroment* env_ = new vt::Enviroment(ctx_);
    env_->execute(text);
    std::string code;
    while( readline(">> ", code ) ) {
        env_->execute( code );
    }

    delete env_;
    delete ctx_;
}

