#include <iostream>
#include <chrono>
#include <tuple>
#include <list>
#include <unistd.h>
#include <sys/wait.h>

#include <tokenizer_combo.hpp>
#include <tensortype_inc.hpp>

#include "memory.hpp"

const size_t MEM_CTX_SIZE = 4 * 1024 * 1024 * 1024l;
const char* batched_text = "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
                           "忽闻水上琵琶声，主人忘归客不发。寻声暗问弹者谁，琵琶声停欲语迟。移船相近邀相见，添酒回灯重开宴。"
                           "千呼万唤始出来，犹抱琵琶半遮面。转轴拨弦三两声，未成曲调先有情。弦弦掩抑声声思，似诉平生不得志。"
                           "低眉信手续续弹，说尽心中无限事。轻拢慢捻抹复挑，初为《霓裳》后《六幺》。大弦嘈嘈如急雨，小弦切切如私语。"
                           "嘈嘈切切错杂弹，大珠小珠落玉盘。间关莺语花底滑，幽咽泉流冰下难。冰泉冷涩弦凝绝，凝绝不通声暂歇。"
                           "别有幽愁暗恨生，此时无声胜有声。银瓶乍破水浆迸，铁骑突出刀枪鸣。曲终收拨当心画，四弦一声如裂帛。"
                           "东船西舫悄无言，唯见江心秋月白。"
                           "沉吟放拨插弦中，整顿衣裳起敛容。自言本是京城女，家在虾蟆陵下住。十三学得琵琶成，名属教坊第一部。"
                           "曲罢曾教善才服，妆成每被秋娘妒。五陵年少争缠头，一曲红绡不知数。钿头银篦击节碎，血色罗裙翻酒污。"
                           "今年欢笑复明年，秋月春风等闲度。弟走从军阿姨死，暮去朝来颜色故。门前冷落鞍马稀，老大嫁作商人妇。"
                           "商人重利轻别离，前月浮梁买茶去。去来江口守空船，绕船月明江水寒。夜深忽梦少年事，梦啼妆泪红阑干。"
                           "我闻琵琶已叹息，又闻此语重唧唧。同是天涯沦落人，相逢何必曾相识！我从去年辞帝京，谪居卧病浔阳城。"
                           "浔阳地僻无音乐，终岁不闻丝竹声。住近湓江地低湿，黄芦苦竹绕宅生。其间旦暮闻何物？杜鹃啼血猿哀鸣。"
                           "春江花朝秋月夜，往往取酒还独倾。岂无山歌与村笛？呕哑嘲哳难为听。今夜闻君琵琶语，如听仙乐耳暂明。"
                           "莫辞更坐弹一曲，为君翻作《琵琶行》。感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。"
                           "座中泣下谁最多？江州司马青衫湿。"
                           "元和十年，予左迁九江郡司马。明年秋，送客湓浦口，闻舟中夜弹琵琶者，听其音，铮铮然有京都声。"
                           "问其人，本长安倡女，尝学琵琶于穆、曹二善才，年长色衰，委身为贾人妇。"
                           "遂命酒，使快弹数曲。曲罢悯然，自叙少小时欢乐事，今漂沦憔悴，转徙于江湖间。"
                           "予出官二年，恬然自安，感斯人言，是夕始觉有迁谪意。"
                           "因为长句，歌以赠之，凡六百一十六言，命曰《琵琶行》。";

struct ChatApplication {
    ChatApplication() {
        tokenizer_ = vt::build_tokenizer_qwen("./qwen.tiktoken");
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

    const int batch = 2;
    const size_t max_output = 128;
    const size_t max_prompt = 7168 - max_output - 8;

    void run() {
        wait_all_ready();

        std::list<std::string>  history;
        history.push_back("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");

        std::string text = batched_text;;
        while(true) {
            std::string new_user =  "<|im_start|>user\n" + text + "<|im_end|>\n";
            history.push_back(new_user);
            history.push_back("<|im_start|>assistant\n");
            std::vector<int> input_tokens = std::move( build_from_history(history) );

            for(int i = 0; input_tokens.size() < max_prompt; i++ ) {
                input_tokens.push_back( input_tokens[i] );
            }

            std::vector<int> id;
            std::vector<int> mask;
            for (size_t i = 0; i < input_tokens.size(); i++ ) {
                id.push_back( input_tokens[i] );
                mask.push_back(1);
            }
            mask.back() = 2;

            std::cout << "########### Prompt Length = " << id.size() << std::endl;

            std::vector<int> ids;
            std::vector<int> masks;
            std::vector<int> nexts;
            std::vector<int> next{0};

            std::vector<int> out_tokens;
            auto start = std::chrono::high_resolution_clock::now();
            for ( int t = 0; t < (int)max_output; t++) {
                ids.clear();
                masks.clear();

                for (int b = 0; b < batch; b ++) {
                    ids.insert(ids.begin(), id.begin(), id.end());
                    masks.insert(masks.begin(), mask.begin(), mask.end());
                }
                nexts.resize(batch, 0);

                int len = id.size();
                write_all(&batch, sizeof(int));
                write_all(&len, sizeof(int));
                write_all(ids.data(), ids.size() * sizeof(int));
                write_all(masks.data(), masks.size() * sizeof(int));

                vt::CollectiveContext::pipe_read(nexts.data(), nexts.size() * sizeof(int));

                /*
                if ( nexts[0] == tokenizer_->token_eos() ) {
                    break;
                }
                */
                out_tokens.push_back(nexts[0]);
                id.push_back(nexts[0]);
                mask.back() = 1;
                mask.push_back(2);

                if ( t == 1 ) {
                    start = std::chrono::high_resolution_clock::now();
                }
                std::string nstr = tokenizer_->decode(nexts[0]);
                printf("%s", nstr.c_str());
                fflush(stdout);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            std::string talk = tokenizer_->decode( out_tokens );
            std::cout << "\n====== Generated " << out_tokens.size() * batch << " total tokens, batch = " << batch << ", using " << duration.count() / 1000.0 << " seconds, ";
            std::cout << (nexts[0] == tokenizer_->token_eos()) << " eos ending ===== " << std::endl;
            //std::cout << talk << std::endl;

            //talk = talk + "<|im_end|>\n";
            //history.push_back(talk);
            break;
        }

        int n = -1;
        write_all(&n, sizeof(int));
    }

    std::vector<int> build_from_history(const std::list<std::string>& history) {
        std::vector<int> all_tokens;
        for ( auto i = history.rbegin(); i != history.rend(); i++) {
            auto one_talk = tokenizer_->encode( *i );
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
        ChatApplication* app = new ChatApplication();
        app->run();
        delete app;

        // wait for all child processes finished
        {
            int status = 0;
            while ( wait(&status) != -1) {
            }
        }
    } else if ( vt::CollectiveContext::pipe_rank == 1) {
        vt::MemoryContext::boot( MEM_CTX_SIZE );
        vt::ComputingContext::boot( 0 );
        vt::Enviroment* env = new vt::Enviroment();
        env->insert_native_word("app.mem", MemoryCounting::creator);

        do_inference(env, dag_file);

        delete env;
        vt::ComputingContext::shutdown();
        vt::MemoryContext::shutdown();
    }
    vt::CollectiveContext::shutdown();
}

