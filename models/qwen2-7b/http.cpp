#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <unordered_map>
#include <sys/wait.h>
#include <locale.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

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

const char* init_cmd = "gpu_init";
const char* main_cmd = "gpu_main";

const size_t max_input = 1024;
const size_t max_output = 1024;

struct ChatApplication {
    ChatApplication(vt::Enviroment* env) : env_(env) {
        tokenizer_ = vt::build_tokenizer_qwen("./qwen.tiktoken", true);
        target_code_ = env_->build(main_cmd);

        svr_ = new httplib::Server();
    }
    ~ChatApplication() {
        svr_->stop();
        delete svr_;

        delete target_code_;
        delete tokenizer_;
    }

    void write_all(const void* buf, size_t nbyte) {
        vt_assert( env_->ctx()->pipe_write(0, buf, nbyte) > 0, "write_all error");
    }

    const char* system_talk =
        "<|im_start|>system\n"
        "你现在开始扮演一个知识小助手。"
        "<|im_end|>\n<|im_start|>user\n";

    void run() {
        svr_->set_mount_point("/", "./www");

        svr_->Post("/chat", [&](const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &content_reader) {
            if (req.is_multipart_form_data()) {
                res.set_content("{'error': 'Don't support multipart post!'}\r\n\r\n", "application/json");
                return;
            }

            std::string body;
            content_reader([&](const char *data, size_t data_length) {
                    body.append(data, data_length);
                    return true;
                });

            std::string result;
            auto s = inference(body, result);
            if ( s == 0 ) {
                result = result + "\r\n\r\n";
                res.set_content(result, "text/plain");
                return;
            }

            res.set_content("{'error': '系统错误!'}\r\n\r\n", "application/json");
        });

        svr_->listen("0.0.0.0", 8888);
    }

    int inference(const std::string& text, std::string& result) {
        std::lock_guard lk(mt_);

        history.push_back(system_talk);

        if ( text == "!" ) {
            history.clear();
            history.push_back(system_talk);
            return 0;
        }

        std::string new_user = text + "<|im_end|>\n";
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
            int len = id.size();
            write_all(id.data(), id.size() * sizeof(int));
            write_all(mask.data(), mask.size() * sizeof(int));

            env_->stack().push_number(1);
            env_->stack().push_number(len);
            env_->run(target_code_);

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

        result = talk;

        talk = talk + "<|im_end|>\n";
        history.push_back(talk);

        return 0;
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
        ::add_history(line);
        free(line);
        return true;
    }



private:
    vt::Tokenizer* tokenizer_;
    vt::Enviroment* env_;
    vt::DaG* target_code_;

    httplib::Server *svr_;

    std::mutex mt_;

    std::list<std::string>  history;
};

int main(int argc, const char* argv[] ) {
    if ( argc < 2 ) {
        std::cout << "usage: ./http [dag_files] " << std::endl;
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

