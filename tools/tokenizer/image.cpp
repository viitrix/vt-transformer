#include <iostream>
#include <chrono>
#include <tuple>
#include <vector>
#include <unistd.h>

#include <tokenizer_combo.hpp>

int main(int argc, const char* argv[]) {
    std::vector<float> out;
    auto loader = vt::build_imageloader_minicpm(argv[1]);

    loader->preprocess(out);
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 32; j++) {
            std::cout << out[32 * 32 * 14 * i + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

