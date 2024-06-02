#pragma once

#ifndef _DAG_MACHINE_H_
#define _DAG_MACHINE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <variant>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace vt {

struct ComputingContext;
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

// Stack & Hash
struct Stack {
    Stack() {}
    ~Stack() {}

    size_t size() {
        return data_.size();
    }
    void clear() {
        data_.clear();
    }

    Cell& top() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        return data_.back();
    }
    Cell pop() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret;
    }
    void drop() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        data_.pop_back();
    }
    void dup() {
        data_.push_back( top() );
    }
    void dup2() {
        auto a = pop();
        auto b = pop();
        for(int i = 0; i < 3; i++) {
            data_.push_back(b);
            data_.push_back(a);
        }
    }
    void swap() {
        auto a = pop();
        auto b = pop();
        data_.push_back(a);
        data_.push_back(b);
    }
    void rot() {
        auto a = pop();
        auto b = pop();
        auto c = pop();
        data_.push_back(b);
        data_.push_back(a);
        data_.push_back(c);
    }
    void rev(int n) {
        if ( n == -1) {
            data_.reverse();
            return;
        }
        if ( n < 0 ) {
           vt_panic("reverse can't support negtive number!");
        }

        std::vector<Cell> temp;
        for(int i = 0; i < n; n++) {
            temp.push_back( pop() );
        }
        for(int i = 0; i < n; n++) {
            data_.push_back( temp[i] );
        }
    }
    double pop_number() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_number();
    }
    std::vector<double> pop_number_list() {
        size_t s = (size_t) pop_number();
        std::vector<double> ret(s, 0.0);
        for (size_t i = 0; i < s ; i++) {
            ret[s - 1 - i] = pop_number();
        }
        return ret;
    }
    const std::string pop_string() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_string();
    }
    tensor_t pop_tensor() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_tensor();
    }
    bool pop_boolean() {
        vt_assert(data_.size() > 0, "Can't access cell from empty stack!");
        auto ret =  data_.back();
        data_.pop_back();
        return ret.as_boolean();
    }
    void push(Cell cell) {
        data_.push_back(cell);
    }
    void push_number(double n) {
        data_.push_back( Cell(n) );
    }
    void push_number_list(std::vector<double>& list) {
        for (size_t i = 0; i < list.size(); i++) {
            push_number( list[i] );
        }
        push_number( list.size() );
    }
    void push_tensor(tensor_t t) {
        data_.push_back( Cell(t) );
    }
    void push_string(const std::string& str) {
        data_.push_back( Cell(str) );
    }

private:
    std::list< Cell> data_;

    friend std::ostream& operator<<(std::ostream& os, Stack& stack);
    friend struct BuiltinOperator;
};
std::ostream& operator<<(std::ostream& os, Stack& stack);

struct Hash {
    using Item = std::variant<double, std::string, tensor_t>;
    Hash() {
    }
    ~Hash() {}

    Item find(const std::string& name) {
        if ( map_.find(name) == map_.end() ) {
            std::cout << "Find: " << name << std::endl;
            vt_panic("Can't find value for name!");
        }
        return map_[name];
    }

    double find_number(const std::string& name) {
        auto item = find(name);
        return std::get<0>(item);
    }

    std::string find_string(const std::string& name) {
        auto item = find(name);
        return std::get<1>(item);
    }

    tensor_t find_tensor(const std::string& name) {
        auto item = find(name);
        return std::get<2>(item);
    }

    void set(const std::string& name, Item item) {
        map_[name] = item;
    }

    void drop(const std::string& name) {
        map_.erase(name);
    }

    static Cell Item2Cell( Item item ) {
        if ( item.index() == 0 ) {
            return Cell( std::get<0>(item) );
        } else if ( item.index() == 1 ) {
            return Cell( std::get<1>(item) );
        }
        return Cell( std::get<2>(item) );
    }

    static Item Cell2Item( Cell& cell ) {
        Item ret;
        if ( cell.type_ == Cell::T_Number ) {
            ret = cell.as_number();
        } else if ( cell.type_ == Cell::T_String ) {
            ret = cell.as_string();
        } else {
            ret = cell.as_tensor();
        }
        return ret;
    }

private:
    std::map<const std::string, Item> map_;
};

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

    static WordCode new_number(double n) {
        WordCode wc;
        wc.type_ = WordCode::Number;
        wc.num_ = n;
        return wc;
    }
    static WordCode new_string(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::String;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_builtin(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::Builtin;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_native(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::Native;
        wc.str_ = v;
        return wc;
    }
    static WordCode new_user(std::string v) {
        WordCode wc;
        wc.type_ = WordCode::User;
        wc.str_ = v;
        return wc;
    }
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

    WordByte(double num) : type_(Number) {
        num_ = num;
    }
    WordByte(const std::string& str) : type_(String) {
        str_ = str;
    }
    WordByte(_WordByteType_ t, size_t i): type_(t) {
        idx_ = i;
    }
    WordByte(_WordByteType_ t): type_(t) {
    }

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
    ~Enviroment() {
    }

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
        return stack_;
    }
    Hash& hash() {
        return hash_;
    }

    ComputingContext* ctx() {
        return ctx_;
    }

private:
    void run_(DaG* dag) {
        auto& binary_ = dag->binary_;
        auto& builtins_ = dag->builtins_;
        auto& natives_ = dag->natives_;

        for ( size_t pc = 0; pc < binary_.size(); ) {
            auto& byte = binary_[pc];
            int steps = 1;
            switch( byte.type_ ) {
                case WordByte::Number:
                    stack_.push_number( byte.num_ );
                    break;

                case WordByte::String:
                    stack_.push_string( byte.str_ );
                    break;

                case WordByte::BuiltinOperator:
                    steps = builtins_[ byte.idx_ ]->run( this );
                    break;

                case WordByte::Native:
                    natives_[ byte.idx_ ]->run( stack_ );
                    break;

                default:
                    vt_panic("Runing binary error can't bere!");
                    break;
            }
            vt_assert( steps > 0, " DAG don't support loop ");
            pc += steps;
        }
    }

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
    Stack stack_;
    Hash hash_;

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

