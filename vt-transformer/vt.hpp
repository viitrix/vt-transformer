#ifndef _VT_HPP_
#define _VT_HPP_

#include <iostream>
#include <random>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <variant>
#include <optional>
#include <algorithm>
#include <cmath>

#define vt_assert(Expr, Msg) \
    vt::_M_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)

#define vt_panic(Msg) \
    vt::_M_Panic(__FILE__, __LINE__, Msg)

#define vt_fatal_error() \
    vt::_M_Panic(__FILE__, __LINE__, "Fatal error, can't be here")

namespace vt {

const unsigned int VERSION = 0x000020;    // v0.2.0

// some common help functions
inline void _M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
    if (!expr) {
        std::cerr << "**Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

inline void _M_Panic(const char* file, int line, const char* msg) {
    std::cerr << "**Panic:\t" << msg << "\n"
        << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
}

struct ComputingContext {
    // something like host:0;cuda:1
    ComputingContext();
    ~ComputingContext() {
        shutdown();
    }

    void* host_workspace;
    size_t workspace_size;
    std::mt19937* rng;

    int      pipe_world;
    int      pipe_rank;
    std::vector<int> pipe_fds;
    void boot_host(const int ranks);
    int pipe_read(void *buf, size_t nbyte);
    int pipe_write(const int n, const void *buf, size_t nbyte);

protected:
    virtual void shutdown();
};

struct TensorType;
using tensor_t = std::shared_ptr<vt::TensorType>;

// target number type
struct Cell {
    enum CellType {
        T_Number,
        T_String,
        T_Tensor,
    };
    const CellType type_;
    std::variant<double, const std::string, tensor_t> v_;

    // constructors
    Cell() : type_(T_Number), v_(0.0f) {}
    Cell(double value): type_(T_Number), v_(value) {}
    Cell(const std::string& value): type_(T_String), v_(value) {}
    Cell(tensor_t value) : type_(T_Tensor), v_(value) {}

    // fast access
    const std::string& as_string() {
        vt_assert(type_ == T_String, "Cell type can't convert to string!");
        return std::get<1>(v_);
    }
    bool as_boolean() {
        vt_assert(type_ == T_Number, "Cell type can't convert to boolean!");
        auto num = std::get<0>(v_);
        if ( num == 0.0) {
            return false;
        }
        return true;
    }
    double as_number() {
        vt_assert(type_ == T_Number, "Cell type can't convert to number!");
        return std::get<0>(v_);
    }
    tensor_t as_tensor() {
        vt_assert(type_ == T_Tensor, "Cell type can't convert to vector!");
        return std::get<2>(v_);
    }
    bool is_number() {
        if ( type_ == T_Number ) {
            return true;
        }
        return false;
    }
    bool is_string() {
        if ( type_ == T_String ) {
            return true;
        }
        return false;
    }
    bool is_tensor() {
        if ( type_ == T_Tensor ) {
            return true;
        }
        return false;
    }
};
std::ostream& operator<<(std::ostream& os, Cell& c);

// Stack & Hash  (defined in vt.cpp)
struct Stack;
struct Hash;

struct WordCode {
    enum {
        Number,
        String,
        Builtin,
        Native,
        User,
    } type_;

    std::string str_;
    double num_;

    static WordCode new_number(double n);
    static WordCode new_string(std::string v);
    static WordCode new_builtin(std::string v);
    static WordCode new_native(std::string v);
    static WordCode new_user(std::string v);
};

struct WordByte {
    enum _WordByteType_ {
        Number,
        String,
        BuiltinOperator,
        Native,
    } type_;

    std::string str_;
    size_t idx_;
    double num_;

    WordByte(double num);
    WordByte(const std::string& str);
    WordByte(_WordByteType_ t, size_t i);
    WordByte(_WordByteType_ t);
};

std::ostream& operator<<(std::ostream& os, const WordCode& c);

// two type local function
struct Enviroment;
struct BuiltinOperator {
    virtual ~BuiltinOperator() {
    }
    virtual int run(Enviroment* env) = 0;
};
struct NativeWord {
    virtual ~NativeWord() {
    }
    virtual void run(Stack& stack) = 0;
};

using NativeCreator = NativeWord* (Enviroment&);
using UserWord = std::vector<WordCode>;
using UserBinary = std::vector<WordByte>;

struct DaG {
    DaG() = delete;
    DaG(Enviroment* env) : env_(env) {}

    ~DaG() {
        for (size_t i = 0; i < natives_.size(); i++) {
            delete natives_[i];
        }
        for (size_t i = 0; i < builtins_.size(); i++) {
            delete builtins_[i];
        }
    }

private:
    // borned from
    const Enviroment* env_;

    // linked and resources
    UserBinary binary_;
    std::vector<NativeWord*> natives_;
    std::vector<BuiltinOperator*> builtins_;

    friend struct Enviroment;
};

struct Enviroment {
    Enviroment(ComputingContext *ctx);
    ~Enviroment();

    void insert_native_word(const std::string& name, NativeCreator* fn) {
        if ( native_words_.find(name) != native_words_.end() ) {
            vt_panic("Can't insert native word with same name!");
        }
        native_words_[name] = fn;
    }

    DaG* build(const std::string& txt) {
        DaG* dag = new DaG(this);

        UserWord myCode = compile(txt);
        linking(*dag, myCode);
        return dag;
    }
    void run(DaG* dag) {
        vt_assert( dag->env_ == this, "Can't be here!");
        run_(dag);
    }

    void execute(const std::string& txt) {
        DaG dag(this);

        UserWord myCode = compile(txt);
        linking(dag, myCode);
        run_(&dag);
    }

    Stack& stack() {
        return *stack_;
    }
    Hash& hash() {
        return *hash_;
    }

    ComputingContext* ctx() {
        return ctx_;
    }

private:
    void run_(DaG* dag);

    NativeWord* create_native(const std::string& name) {
        if ( native_words_.find(name) != native_words_.end() ) {
            return native_words_[name](*this);
        }
        vt_panic("Call a un registered native word!");
        return nullptr;
    }

    UserWord& get_user(const std::string& name) {
        if ( user_words_.find(name) == user_words_.end() ) {
            vt_panic("Call a un registered native word!");
        }
        return user_words_[name];
    }
    UserWord compile(const std::string& txt);
    void linking(DaG& dag, UserWord& word);

    void load_base_words();
private:
    // compiled
    std::map<std::string, UserWord> user_words_;
    std::map<std::string, NativeCreator*> native_words_;

    // runtime
    std::unique_ptr<Stack> stack_;
    std::unique_ptr<Hash> hash_;

    // global
    ComputingContext *ctx_;
};

#define NWORD_CREATOR_DEFINE_LR(CLS)         \
static NativeWord* creator(vt::Enviroment& env) {   \
    vt::NativeWord* wd = new CLS();                \
    return wd;                                 \
}

#define NWORD_CREATOR_DEFINE_CTX(CLS)                \
vt::ComputingContext* ctx_;                             \
CLS(vt::Enviroment& env) : ctx_(env.ctx()) {   \
}                                                   \
static NativeWord* creator(vt::Enviroment& env) {   \
    vt::NativeWord* wd = new CLS(env);                \
    return wd;                                 \
}

} // end of namespace
#endif
