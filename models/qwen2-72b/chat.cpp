#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <sys/wait.h>
#include <locale.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#include "memory.hpp"

const char* init_cmd = "gpu_init";
const char* main_cmd = "gpu_main";
const size_t max_input = 512;
const size_t max_output = 512;

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
    ChatApplication(vt::Enviroment* env) : env_(env) {
        tokenizer_ = vt::build_tokenizer_qwen("./qwen.tiktoken");
        main_code_ = env_->build(main_cmd);
    }
    ~ChatApplication() {
        delete tokenizer_;
        delete main_code_;
    }

    void write_all(const void* buf, size_t nbyte) {
        vt_assert( env_->ctx()->pipe_write(0, buf, nbyte) > 0, "write_all error");
    }

    void run() {

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
                write_all(id.data(), id.size() * sizeof(int));
                write_all(mask.data(), mask.size() * sizeof(int));

                env_->stack().push_number(batch);
                env_->stack().push_number(len);
                env_->run(main_code_);

                next = -1;
                env_->ctx()->pipe_read(&next, sizeof(int));
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
        char *line = ::readline(prop.c_str());

        if ( line == NULL ) {
            return false;
        }
        code = line;
        free(line);
        return true;
    }

private:
    vt::Tokenizer* tokenizer_;
    vt::Enviroment* env_;
    vt::DaG* main_code_;
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

#ifdef _USING_DEVICE_HIP_
        env->stack().push_string("hip");
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


int main(int argc, const char* argv[] ) {
    setlocale(LC_ALL, "");
    if ( argc < 2 ) {
        std::cout << "usage: ./chat [dag_files] " << std::endl;
        return -1;
    }
    vt::ComputingContext* ctx = new vt::ComputingContext();
    ctx->boot_host(0);
#ifdef _USING_DEVICE_CUDA_
    ctx->boot_cuda(0);
#endif
    vt::Enviroment* env = new vt::Enviroment(ctx);
    env->insert_native_word("app.mem", MemoryCounting::creator);
    env->insert_native_word("app.align", MemoryAlign::creator);

    // load whole code
    {
        std::string all_code;
        for(int i = 1; i < argc; i++) {
            const char* dag_file = argv[i];
            all_code += fileToString(dag_file);
        }

        vt::DaG* init_bin = env->build(all_code);
#ifdef _USING_DEVICE_CUDA_
        env->stack().push_string("cuda");
#endif
        env->run(init_bin);
        delete init_bin;
    }
    env->execute(init_cmd);

    ChatApplication* app = new ChatApplication(env);
    app->run();

    delete app;
    delete env;
    delete ctx;
}


