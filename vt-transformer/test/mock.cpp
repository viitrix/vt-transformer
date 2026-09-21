#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "scheduler.hpp"
#include "mock_backend.hpp"
#include "mock_engine.hpp"

#include "vt_cuda.hpp"

// 测试使用的固定配置, 与 MockEngine 保持一致
static constexpr std::size_t kMaxSeqLen = 1024;
static constexpr int        kMaxReq    = 4;

using Token = int32_t;
using Index = int32_t;
using TestScheduler = vt::Scheduler<kMaxSeqLen, kMaxReq>;
using TestBatchTable = vt::BatchTable<kMaxReq, Token, Index>;

// 读取整个文件到字符串; 失败直接 panic
std::string read_file(const char* path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    vt_assert(f.good() && !ss.str().empty(), "read_file: cannot read file");
    return ss.str();
}

void initParameter(vt::Enviroment* env) {
    constexpr std::size_t kFlagsSize = 16;
    size_t tok_table_size = sizeof(Token) * kMaxReq * kMaxSeqLen;
    size_t idx_table_size = sizeof(Index) * kMaxReq * kMaxSeqLen;
    size_t flags_table_size = sizeof(int32_t) * kMaxReq * kFlagsSize;
    size_t batch_table_size = sizeof( vt::BatchTable<kMaxReq, Token, Index> ) * 2;

    env->stack().push_number( batch_table_size);
    env->stack().push_number( flags_table_size);
    env->stack().push_number( idx_table_size);
    env->stack().push_number( tok_table_size);
}

int main() {
    vt::Enviroment* env = vt::create_vt_cuda(0);
    initParameter(env);
    env->execute( read_file("./init.dag") );

    // 从 VT 虚拟机中提取 init.dag 分配的 Slot/Batch 表 (host + device),
    auto* tok_host = static_cast<Token*>( env->hash().find_tensor("tok_table_host")->data() );
    auto* idx_host = static_cast<Index*>( env->hash().find_tensor("idx_table_host")->data() );
    auto* tok_dev  = static_cast<Token*>( env->hash().find_tensor("tok_table")->data() );
    auto* idx_dev  = static_cast<Index*>( env->hash().find_tensor("idx_table")->data() );
    auto* flags_host = static_cast<int32_t*>( env->hash().find_tensor("flags_table_host")->data() );
    auto* flags_dev  = static_cast<int32_t*>( env->hash().find_tensor("flags_table")->data() );
    auto* batch_host = static_cast<TestBatchTable*>( env->hash().find_tensor("batch_table_host")->data() );
    auto* batch_dev  = static_cast<TestBatchTable*>( env->hash().find_tensor("batch_table")->data() );

    {
        vt::MockEngine engine(env);
        vt::MockBackend backend;

        TestScheduler sched(tok_host, idx_host, tok_dev, idx_dev,
                            flags_host, flags_dev,
                            batch_host, batch_dev, &engine, &backend);
        sched.main_overlap_loop();
    }

    delete env;
    return 0;
}
