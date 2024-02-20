#include <iostream>
#include <chrono>
#include <tuple>
#include <vector>
#include <unistd.h>

#include <tokenizer_combo.hpp>

int main(int argc, const char* argv[]) {
    std::vector<uint8_t> rgbdata;
    std::string file = argv[1];
    std::vector<uint8_t> data;

    vt::load_image_rgb_plane(file, data);

    std::cout << data.size() << std::endl;
}

