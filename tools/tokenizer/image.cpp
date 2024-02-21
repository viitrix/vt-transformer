#include <iostream>
#include <chrono>
#include <tuple>
#include <vector>
#include <unistd.h>

#include <tokenizer_combo.hpp>

int main(int argc, const char* argv[]) {
    std::vector<float> out;
    auto loader = vt::build_imageloader_qwen(argv[1]);

    loader->preprocess(out);
    for(int i = 0; i < 8; i++) {
        std::cout << out[448 * 448 * 2 + i] << " ";
        //std::cout << out[ i] << " ";
    }
    std::cout << std::endl;
}

