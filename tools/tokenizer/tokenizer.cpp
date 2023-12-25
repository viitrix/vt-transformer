#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <tokenizer_combo.hpp>

int main(int argc, const char* argv[]) {
    auto tokenizer = vt::build_tokenizer_qwen("../../models/qwen-7b/qwen.tiktoken");

    std::string text = argv[1];
    auto tokens = tokenizer->encode(text);
    for(auto& v : tokens) {
        std::cout << v << "\t |" << tokenizer->decode(v)  << "|" << std::endl;
    }
    std::vector<int> ids;
    for ( size_t i = 0; i < tokens.size(); i++) {
        ids.push_back( tokens[i] );
    }
    std::cout << tokenizer->decode( ids) << std::endl;

    delete tokenizer;
    return 0;
}
