#include <iostream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <sys/wait.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#include "memory.hpp"

const size_t MEM_CTX_SIZE = 16 * 1024 * 1024 * 1024l;

struct ChatApplication {
    ChatApplication() {
        tokenizer_ = vt::build_tokenizer_llama3("./llama3.tiktoken");
    }
    ~ChatApplication() {
        delete tokenizer_;
    }

    void write_all(const void* buf, size_t nbyte) {
        vt_assert( vt::CollectiveContext::pipe_write(1, buf, nbyte) > 0, "write_all error");
    }

    void wait_all_ready() {
        int dummy = -1;
        vt::CollectiveContext::pipe_read(&dummy, sizeof(int));
        vt_assert(dummy == 1, "wait_all_ready error");
    }

    const size_t max_input = 512;
    const size_t max_output = 512;

    //
    // messages = [{
    //          "role": "user", "content": "你是一名Java老师，请出一道关于Java异常的编程练习题"
    //     },]
    //
    //     talk = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True);
    //
    // '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是一名Java老师，请出一道关于Java异常的编程练习题<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    //
    void run() {
        const char* begin_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>";
        std::list<std::string>  history;
        history.push_back(begin_text);

        std::string text;
        while( readline(">> ", text) ) {
            if ( text.size() <= 0 ) {
                continue;
            }
            if ( text == "!" ) {
                history.clear();
                history.push_back(begin_text);
                continue;
            }

            std::string new_user =  "<|start_header_id|>user|end_header_id|>\n\n" + text + "<|eot_id|>";
            history.push_back(new_user);
            history.push_back("<|start_header_id|>assistant<|end_header_id|>\n\n");
            std::vector<int> input_tokens = build_from_history(history);

            for(int i = 0; i < (int)input_tokens.size(); i++) {
                std::cout << input_tokens[i] << " ";
            }
            std::cout << std::endl;


            std::vector<int> id;
            std::vector<int> mask;
            for (size_t i = 0; i < input_tokens.size(); i++ ) {
                id.push_back( input_tokens[i] );
                mask.push_back(1);
            }
            mask.back() = 2;

            std::vector<int> out_tokens;

            auto start = std::chrono::high_resolution_clock::now();
            int next = -1;
            for ( int t = 0; t < (int)max_output; t++) {
                int batch = 1;
                int len = id.size();
                write_all(&batch, sizeof(int));
                write_all(&len, sizeof(int));
                write_all(id.data(), id.size() * sizeof(int));
                write_all(mask.data(), mask.size() * sizeof(int));

                next = -1;
                vt::CollectiveContext::pipe_read(&next, sizeof(int));
                if ( next == tokenizer_->token_eos() ) {
                    break;
                }
                out_tokens.push_back(next);
                id.push_back(next);
                mask.back() = 1;
                mask.push_back(2);

                std::string nstr = tokenizer_->decode(next);
                //std::cout << nstr << std::flush;
                printf("%s", nstr.c_str());
                fflush(stdout);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            std::string talk = tokenizer_->decode( out_tokens );
            std::cout << "\n====== Generated " << out_tokens.size() << " tokens, using " << duration.count() / 1000.0 << " seconds, ";
            std::cout << (next == tokenizer_->token_eos()) << " eos ending ===== " << std::endl;
            //std::cout << talk << std::endl;

            talk = talk + "<|eot_id|>\n";
            history.push_back(talk);
        }

        int n = -1;
        write_all(&n, sizeof(int));
    }

    std::vector<int> build_from_history(const std::list<std::string>& history) {
        std::vector<int> all_tokens;
        for ( auto i = history.rbegin(); i != history.rend(); i++) {
            auto one_talk = tokenizer_->encode( *i );
            if ( one_talk.size() + all_tokens.size() >= max_input ) {
                break;
            }
            all_tokens.insert( all_tokens.end(), one_talk.rbegin(), one_talk.rend() );
        }
        std::reverse(all_tokens.begin(), all_tokens.end());
        return all_tokens;
    }

    bool readline(const std::string& prop, std::string& code) {
        std::cout << prop << std::flush;
        if ( std::getline(std::cin, code) ) {
            return true;
        }
        return false;
    }

private:
    vt::Tokenizer* tokenizer_;
};

int main(int argc, char* argv[] ) {
    vt::CollectiveContext::boot_pipe(1);

    if ( vt::CollectiveContext::pipe_rank == 0) {
        ChatApplication* app = new ChatApplication();
        app->run();
        delete app;
    } else if ( vt::CollectiveContext::pipe_rank == 1) {

    }
    vt::CollectiveContext::shutdown();
}

