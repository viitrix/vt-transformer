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
const char* batched_text = R"___(下面请仔细阅读。
　　孔雀东南飞，五里一徘徊。
　　“十三能织素，十四学裁衣。十五弹箜篌，十六诵诗书。十七为君妇，心中常苦悲。君既为府吏，守节情不移。贱妾留空房，相见常日稀。鸡鸣入机织，夜夜不得息。三日断五匹，大人故嫌迟。非为织作迟，君家妇难为！妾不堪驱使，徒留无所施。便可白公姥，及时相遣归。”
　　府吏得闻之，堂上启阿母：“儿已薄禄相，幸复得此妇。结发同枕席，黄泉共为友。共事二三年，始尔未为久。女行无偏斜，何意致不厚。”
　　阿母谓府吏：“何乃太区区！此妇无礼节，举动自专由。吾意久怀忿，汝岂得自由！东家有贤女，自名秦罗敷。可怜体无比，阿母为汝求。便可速遣之，遣去慎莫留！”
　　府吏长跪告：“伏惟启阿母。今若遣此妇，终老不复取！”
　　阿母得闻之，槌床便大怒：“小子无所畏，何敢助妇语！吾已失恩义，会不相从许！”
　　府吏默无声，再拜还入户。举言谓新妇，哽咽不能语：“我自不驱卿，逼迫有阿母。卿但暂还家，吾今且报府。不久当归还，还必相迎取。以此下心意，慎勿违吾语。”
　　新妇谓府吏：“勿复重纷纭。往昔初阳岁，谢家来贵门。奉事循公姥，进止敢自专？昼夜勤作息，伶俜萦苦辛。谓言无罪过，供养卒大恩；仍更被驱遣，何言复来还！妾有绣腰襦，葳蕤自生光；红罗复斗帐，四角垂香囊；箱帘六七十，绿碧青丝绳，物物各自异，种种在其中。人贱物亦鄙，不足迎后人，留待作遗施，于今无会因。时时为安慰，久久莫相忘！”
　　鸡鸣外欲曙，新妇起严妆。著我绣夹裙，事事四五通。足下蹑丝履，头上玳瑁光。腰若流纨素，耳著明月珰。指如削葱根，口如含朱丹。纤纤作细步，精妙世无双。
　　上堂拜阿母，阿母怒不止。“昔作女儿时，生小出野里。本自无教训，兼愧贵家子。受母钱帛多，不堪母驱使。今日还家去，念母劳家里。”却与小姑别，泪落连珠子。“新妇初来时，小姑始扶床；今日被驱遣，小姑如我长。勤心养公姥，好自相扶将。初七及下九，嬉戏莫相忘。”出门登车去，涕落百余行。
　　府吏马在前，新妇车在后。隐隐何甸甸，俱会大道口。下马入车中，低头共耳语：“誓不相隔卿，且暂还家去。吾今且赴府，不久当还归。誓天不相负！”
　　新妇谓府吏：“感君区区怀！君既若见录，不久望君来。君当作磐石，妾当作蒲苇。蒲苇纫如丝，磐石无转移。我有亲父兄，性行暴如雷，恐不任我意，逆以煎我怀。”举手长劳劳，二情同依依 。
　　入门上家堂，进退无颜仪。阿母大拊掌，不图子自归：“十三教汝织，十四能裁衣，十五弹箜篌，十六知礼仪，十七遣汝嫁，谓言无誓违。汝今何罪过，不迎而自归？”兰芝惭阿母：“儿实无罪过。”阿母大悲摧。
　　还家十余日，县令遣媒来。云有第三郎，窈窕世无双。年始十八九，便言多令才。
　　阿母谓阿女：“汝可去应之。”
　　阿女含泪答：“兰芝初还时，府吏见丁宁，结誓不别离。今日违情义，恐此事非奇。自可断来信，徐徐更谓之。”
　　阿母白媒人：“贫贱有此女，始适还家门。不堪吏人妇，岂合令郎君？幸可广问讯，不得便相许。”
　　媒人去数日，寻遣丞请还，说有兰家女，承籍有宦官。云有第五郎，娇逸未有婚。遣丞为媒人，主簿通语言。直说太守家，有此令郎君，既欲结大义，故遣来贵门。
　　阿母谢媒人：“女子先有誓，老姥岂敢言！”
　　阿兄得闻之，怅然心中烦。举言谓阿妹：“作计何不量！先嫁得府吏，后嫁得郎君。否泰如天地，足以荣汝身。不嫁义郎体，其往欲何云？”
　　兰芝仰头答：“理实如兄言。谢家事夫婿，中道还兄门。处分适兄意，那得自任专！虽与府吏要，渠会永无缘。登即相许和，便可作婚姻。”媒人下床去。诺诺复尔尔。还部白府君：“下官奉使命，言谈大有缘。”府君得闻之，心中大欢喜。视历复开书，便利此月内，六合正相应。良吉三十日，今已二十七，卿可去成婚。交语速装束，络绎如浮云。青雀白鹄舫，四角龙子幡。婀娜随风转，金车玉作轮。踯躅青骢马，流苏金镂鞍。赍钱三百万，皆用青丝穿。杂彩三百匹，交广市鲑珍。从人四五百，郁郁登郡门。
　　阿母谓阿女：“适得府君书，明日来迎汝。何不作衣裳？莫令事不举！”
　　阿女默无声，手巾掩口啼，泪落便如泻。移我琉璃榻，出置前窗下。左手持刀尺，右手执绫罗。朝成绣夹裙，晚成单罗衫。晻晻日欲暝，愁思出门啼。
　　府吏闻此变，因求假暂归。未至二三里，摧藏马悲哀。新妇识马声，蹑履相逢迎。怅然遥相望，知是故人来。举手拍马鞍，嗟叹使心伤：“自君别我后，人事不可量。果不如先愿，又非君所详。我有亲父母，逼迫兼弟兄。以我应他人，君还何所望！”
　　府吏谓新妇：“贺卿得高迁！磐石方且厚，可以卒千年；蒲苇一时纫，便作旦夕间。卿当日胜贵，吾独向黄泉！”
　　新妇谓府吏：“何意出此言！同是被逼迫，君尔妾亦然。黄泉下相见，勿违今日言！”执手分道去，各各还家门。生人作死别，恨恨那可论？念与世间辞，千万不复全！
　　府吏还家去，上堂拜阿母：“今日大风寒，寒风摧树木，严霜结庭兰。儿今日冥冥，令母在后单。故作不良计，勿复怨鬼神！命如南山石，四体康且直！”
　　阿母得闻之，零泪应声落：“汝是大家子，仕宦于台阁。慎勿为妇死，贵贱情何薄！东家有贤女，窈窕艳城郭，阿母为汝求，便复在旦夕。”
　　府吏再拜还，长叹空房中，作计乃尔立。转头向户里，渐见愁煎迫。
　　其日牛马嘶，新妇入青庐。奄奄黄昏后，寂寂人定初。“我命绝今日，魂去尸长留！”揽裙脱丝履，举身赴清池。
　　府吏闻此事，心知长别离。徘徊庭树下，自挂东南枝。
　　两家求合葬，合葬华山傍。东西植松柏，左右种梧桐。枝枝相覆盖，叶叶相交通。中有双飞鸟，自名为鸳鸯。仰头相向鸣，夜夜达五更。行人驻足听，寡妇起彷徨。多谢后世人，戒之慎勿忘。
请根据上面的材料，文章主要说了什么？)___";

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

    const int batch = 8;
    const int max_output = 128;

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
                if ( nexts[0] == tokenizer_->token_eos() ) {
                    break;
                }
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
#ifdef _USING_DEVICE_CUDA_
        vt::ComputingContext::boot_cuda( 0 );
#endif
#ifdef _USING_DEVICE_DCU_
        vt::ComputingContext::boot_dcu( 0 );
#endif
        vt::Enviroment* env = new vt::Enviroment();
        env->insert_native_word("app.mem", MemoryCounting::creator);

        do_inference(env, dag_file);

        delete env;
        vt::ComputingContext::shutdown();
        vt::MemoryContext::shutdown();
    }
    vt::CollectiveContext::shutdown();
}

