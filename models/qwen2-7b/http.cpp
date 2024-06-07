#include <iostream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <sys/wait.h>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <mutex>
#include <sstream>
#include <thread>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#include "memory.hpp"

const size_t MEM_CTX_SIZE = 4 * 1024 * 1024 * 1024l;

struct ChatApplication {
    ChatApplication() {
        tokenizer_ = vt::build_tokenizer_qwen("./qwen.tiktoken");
        inferencing_ = false;
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

    const size_t max_input = 2048;
    const size_t max_output = 1024;

    static void server(ChatApplication* app, httplib::Server& svr) {
        svr.Post("/chat", [&](const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &content_reader) {
            if ( app->inferencing_ == true ) {
                res.set_content("{'error': 'System is busy!'}\r\n\r\n", "application/json");
                return;
            }
            if (req.is_multipart_form_data()) {
                res.set_content("{'error': 'Don't support multipart post!'}\r\n\r\n", "application/json");
                return;
            }
            if (req.has_param("reset_cache")) {
                if ( req.get_param_value("reset_cache") == "true" ) {
                    app->writeline("!");
                }
            }

            std::string body;
            content_reader([&](const char *data, size_t data_length) {
                    body.append(data, data_length);
                    return true;
                });

            app->writeline(body);

            res.set_chunked_content_provider("text/event-stream", [&](size_t offset, httplib::DataSink &sink) {
                    std::string one_token = app->readtoken();
                    if ( one_token == "" ) {
                        sink.write("\r\n\r\n", 4);
                        sink.done();
                        return true;
                    }
                    sink.write(one_token.data(), one_token.size());
                    return true;
                });
        });
    }

    void run() {
        wait_all_ready();

        std::list<std::string>  history;
        history.push_back("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");

        std::string text;
        while( readline(text) ) {
            if ( text.size() <= 0 ) {
                continue;
            }
            if ( text == "!" ) {
                history.clear();
                history.push_back("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
                continue;
            }

            inferencing_ = true;

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
                vt::CollectiveContext::pipe_read(&next, sizeof(int));
                if ( next == tokenizer_->token_eos() ) {
                    break;
                }
                out_tokens.push_back(next);
                id.push_back(next);
                mask.back() = 1;
                mask.push_back(2);

                std::string nstr = tokenizer_->decode(next);

                writetoken(nstr);
                printf("%s", nstr.c_str());
                fflush(stdout);

            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            std::string talk = tokenizer_->decode( out_tokens );
            std::cout << "\n====== Generated " << out_tokens.size() << " tokens, using " << duration.count() / 1000.0 << " seconds, ";
            std::cout << (next == tokenizer_->token_eos()) << " eos ending ===== " << std::endl;
            // std::cout << talk << std::endl;

            writetoken("");
            inferencing_ = false;

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

    void writetoken(std::string token) {
        std::lock_guard lk(mt_);
        omq_.push(token);
        ocv_.notify_all();
    }

    std::string readtoken() {
        std::unique_lock<std::mutex> lk(mt_);
        ocv_.wait(lk);

        auto token = omq_.front();
        omq_.pop();
        return token;
    }

    void writeline(std::string text) {
        std::lock_guard lk(mt_);
        imq_.push(text);
        icv_.notify_all();
    }

    bool readline(std::string& code) {
        std::unique_lock<std::mutex> lk(mt_);
        icv_.wait(lk);

        code = imq_.front();
        imq_.pop();
        if ( code == "" ) {
            return false;
        }
        return true;
    }

private:
    std::mutex mt_;
    bool inferencing_;

    std::queue<std::string> imq_;
    std::condition_variable icv_;
    std::queue<std::string> omq_;
    std::condition_variable ocv_;

    vt::Tokenizer* tokenizer_;
};

void do_inference(vt::Enviroment* env, const char* dag_file) {
    const char* init_cmd = "gpu_init";
    const char* main_cmd = "gpu_main";
    {
        std::string all_code = vt::fileToString(dag_file);

        vt::DaG* init_bin = env->build(all_code);
#ifdef _USING_DEVICE_CUDA_
        env->stack().push_string("cuda");
#endif
#ifdef _USING_DEVICE_DCU_
        env->stack().push_string("dcu");
#endif
#ifdef _USING_DEVICE_COREX_
        env->stack().push_string("corex");
#endif
        env->run(init_bin);
        delete init_bin;
    }
    env->execute(init_cmd);

    int ok = 1;
    vt_assert( vt::CollectiveContext::pipe_write(0, &ok, sizeof(int)) > 0, "pipe_write error");

    vt::DaG* target_cmd = env->build(main_cmd);

    for (;;) {
        int batches = -1;
        int id = -1;
        vt::CollectiveContext::pipe_read(&batches, sizeof(int));;
        if ( batches <= 0) {
            break;
        }

        vt::CollectiveContext::pipe_read(&id, sizeof(int));;
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
    const char* dag_file = argv[1];
    vt::CollectiveContext::boot_pipe(1);

    if ( vt::CollectiveContext::pipe_rank == 0) {
        httplib::Server svr;

        ChatApplication* app = new ChatApplication();

        std::thread tserver([&] {
                ChatApplication::server(app, svr);
                svr.listen("localhost", 1234);
            });

        app->run();

        // close threads
        svr.stop();
        tserver.join();

        delete app;
        // wait for all child processes finished
        {
            int status = 0;
            while ( wait(&status) != -1) {
            }
        }
    } else if ( vt::CollectiveContext::pipe_rank == 1) {
        vt::MemoryContext::boot( MEM_CTX_SIZE );
#ifdef _USING_DEVICE_CUDA_
        vt::ComputingContext::boot_cuda( 0 );
#endif
#ifdef _USING_DEVICE_COREX_
        vt::ComputingContext::boot_corex( 0 );
#endif

        vt::Enviroment* env = new vt::Enviroment();
        env->insert_native_word("app.mem", MemoryCounting::creator);
        env->insert_native_word("app.align", MemoryAlign::creator);

        do_inference(env, dag_file);

        delete env;
        vt::ComputingContext::shutdown();
        vt::MemoryContext::shutdown();
    }
    vt::CollectiveContext::shutdown();
}

