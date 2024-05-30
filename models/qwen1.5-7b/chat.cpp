#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <sys/wait.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#include "memory.hpp"

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

struct ChatApplication {
    ChatApplication(vt::ComputingContext* ctx): ctx_(ctx) {
        tokenizer_ = vt::build_tokenizer_qwen("./qwen.tiktoken");
    }
    ~ChatApplication() {
        delete tokenizer_;
    }

    void write_all(const void* buf, size_t nbyte) {
        vt_assert( ctx_->pipe_write(1, buf, nbyte) > 0, "write_all error");
    }

    void wait_all_ready() {
        int dummy = -1;
        ctx_->pipe_read(&dummy, sizeof(int));
        vt_assert(dummy == 1, "wait_all_ready error");
    }

    const size_t max_input = 512;
    const size_t max_output = 512;

    void run() {
        wait_all_ready();

        std::list<std::string>  history;
        history.push_back("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");

        std::string text;
        while( readline(">> ", text) ) {
            if ( text.size() <= 0 ) {
                continue;
            }
            if ( text == "!" ) {
                history.clear();
                history.push_back("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
                continue;
            }

            std::string new_user =  "<|im_start|>user\n" + text + "<|im_end|>\n";
            history.push_back(new_user);
            history.push_back("<|im_start|>assistant\n");
            std::vector<int> input_tokens = std::move( build_from_history(history) );

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
                ctx_->pipe_read(&next, sizeof(int));
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

            talk = talk + "<|im_end|>\n";
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
    vt::ComputingContext* ctx_;
};

void do_inference(vt::Enviroment* env, const char* dag_file) {
    const char* init_cmd = "gpu_init";
    const char* main_cmd = "gpu_main";
    {
        std::string all_code = fileToString(dag_file);

        vt::DaG* init_bin = env->build(all_code);
#ifdef _USING_DEVICE_CUDA_
        env->stack().push_string("cuda");
#endif
        env->run(init_bin);
        delete init_bin;
    }

    env->execute(init_cmd);

    int ok = 1;
    vt_assert( env->ctx()->pipe_write(0, &ok, sizeof(int)) > 0, "pipe_write error");

    vt::DaG* target_cmd = env->build(main_cmd);

    for (;;) {
        int batches = -1;
        int id = -1;
        env->ctx()->pipe_read(&batches, sizeof(int));;
        if ( batches <= 0) {
            break;
        }

        env->ctx()->pipe_read(&id, sizeof(int));;
        vt_assert(id > 0, "tokens can't must more than zero!");

        env->stack().push_number(batches);
        env->stack().push_number(id);
        env->run(target_cmd);
    }

    delete target_cmd;
}



int main(int argc, char* argv[] ) {
    if ( argc < 2 ) {
        std::cout << "usage: ./chat [dag_file] " << std::endl;
        return -1;
    }
    vt::ComputingContext* ctx = new vt::ComputingContext();
    ctx->boot_host(1);

    if ( ctx->pipe_rank == 0) {
        ChatApplication* app = new ChatApplication(ctx);
        app->run();

        delete app;
        delete ctx;

        // wait for all child processes finished
        {
            int status = 0;
            while ( wait(&status) != -1) {
            }
        }
    } else if ( ctx->pipe_rank == 1) {
        ctx->boot_cuda(0);

        vt::Enviroment* env = new vt::Enviroment(ctx);
        env->insert_native_word("app.mem", MemoryCounting::creator);
        env->insert_native_word("app.align", MemoryAlign::creator);
        do_inference(env, argv[1]);

        delete env;
        delete ctx;
    }
}

