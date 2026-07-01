/*
 * merged from: context.cpp, dag.cpp
 */
#include <time.h>
#include <unistd.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <regex>

#include "tensortype.hpp"
#include "vt.hpp"

namespace vt {

// ===================================================================
// ComputingContext  (from context.cpp)
// ===================================================================

const size_t DEFAULT_WORKSPACE_SIZE = 1024 * 1024 * 16;

ComputingContext::ComputingContext() {
    // basic init
    workspace_size = DEFAULT_WORKSPACE_SIZE;
    host_workspace = nullptr;
    pipe_world = 0;
    pipe_rank = -1;
}

void ComputingContext::boot_host(int rks) {
    pipe_world = rks + 1;
    pipe_rank = 0;
    pipe_fds.resize((rks + 1) * 2, -1);
    for (int i = 0; i < rks + 1; i++) {
        int* fds = pipe_fds.data() + i * 2;
        vt_assert( pipe(fds) >= 0, "Can't create pipe between parent and child process!");
    }

    for (int i = 0; i < rks; i++) {
        int n = fork();
        if ( n == 0 ) {
            pipe_rank = i + 1;
            break;
        }
    }

    host_workspace = malloc( workspace_size );
    rng = new std::mt19937(1979);
}

void ComputingContext::shutdown() {
    // host shutdown
    for (size_t i = 0; i < pipe_fds.size(); i++) {
        close( pipe_fds[i]);
    }
    free(host_workspace);
    delete rng;
}

int ComputingContext::pipe_write(const int n, const void *buf, size_t nbyte) {
    if ( pipe_fds.size() == 0) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[n * 2 + 1];
    return write(fd, buf, nbyte);
}

int ComputingContext::pipe_read(void *buf, size_t nbyte) {
    if ( pipe_fds.size() == 0) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[pipe_rank * 2 + 0];
    return read(fd, buf, nbyte);
}

// ===================================================================
// Stack & Hash  (moved here from vt.hpp)
// ===================================================================

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

// ===================================================================
// WordCode / WordByte  (moved here from vt.hpp)
// ===================================================================

WordCode WordCode::new_number(double n) {
    WordCode wc;
    wc.type_ = WordCode::Number;
    wc.num_ = n;
    return wc;
}
WordCode WordCode::new_string(std::string v) {
    WordCode wc;
    wc.type_ = WordCode::String;
    wc.str_ = v;
    return wc;
}
WordCode WordCode::new_builtin(std::string v) {
    WordCode wc;
    wc.type_ = WordCode::Builtin;
    wc.str_ = v;
    return wc;
}
WordCode WordCode::new_native(std::string v) {
    WordCode wc;
    wc.type_ = WordCode::Native;
    wc.str_ = v;
    return wc;
}
WordCode WordCode::new_user(std::string v) {
    WordCode wc;
    wc.type_ = WordCode::User;
    wc.str_ = v;
    return wc;
}

WordByte::WordByte(double num) : type_(Number) {
    num_ = num;
}
WordByte::WordByte(const std::string& str) : type_(String) {
    str_ = str;
}
WordByte::WordByte(_WordByteType_ t, size_t i): type_(t) {
    idx_ = i;
}
WordByte::WordByte(_WordByteType_ t): type_(t) {
}

// ===================================================================
// DaG machine  (from dag.cpp)
// ===================================================================

std::ostream& operator<<(std::ostream& os, Cell& c) {
    if ( c.type_ == Cell::T_String ) {
        os << "S:" << c.as_string();
    } else if ( c.type_ == Cell::T_Number ) {
        os << "N:" << std::fixed << c.as_number();
    } else {
        auto t = c.as_tensor();
        if ( t != nullptr ) {
            os << "T: (" << t->to_string() << " )";
        } else {
            os << "T: (null)";
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const WordCode& c) {
    if ( c.type_ == WordCode::String ) {
        os << "S:" << c.str_;
    } else if ( c.type_ == WordCode::Number ) {
        os << "N:" << c.num_;
    } else if ( c.type_ == WordCode::Builtin ) {
        os << "B:" << c.str_;
    } else if ( c.type_ == WordCode::Native ) {
        os << "NA:" << c.str_;
    } else {
        os << "U:" << c.str_;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Stack& stack) {
    int i = 0;
    os << "----STACK(" << stack.size() << ")----" << std::endl;
    for (auto it = stack.data_.begin(); it != stack.data_.end(); it++, i++) {
        os << "==>" << i << " " << *it << std::endl;
    }
    os << "----" << std::endl;
    return os;
}

UserWord Enviroment::compile(const std::string& txt) {
    struct _ {
        static bool parse_number(const std::string& token, double& value) {
            if (isdigit(token.at(0)) || (token.at(0) == '-' && token.length() >= 2 && isdigit(token.at(1)))) {
                if (token.find('.') != std::string::npos || token.find('e') != std::string::npos) { // double
                    value = atof(token.c_str());
                } else {
                    value = atol(token.c_str());
                }
                return true;
            }
            return false;
        }

        static void tokenize_line(std::string const &str_line, std::vector<std::string> &out) {
            std::string state = "SYMBOL";
            std::string current_str = "";

            for (size_t i = 0; i < str_line.size(); i++) {
                char cur = str_line[i];
                if ( state == "SYMBOL") {
                    if ( cur == ' ' || cur == '('  || cur == ')' || cur == '{' || cur == '}' ) {
                        // ending of a symbol
                        if (current_str != "") {
                            out.push_back(current_str);
                            current_str = "";
                        }
                        continue;
                    }
                    if ( cur == '[' || cur == ']' ) {
                        // ending of a symbol
                        if (current_str != "") {
                            out.push_back(current_str);
                            current_str = "";
                        }
                        out.push_back( std::string(1, cur) );
                        continue;
                    }

                    if ( cur == '"' ) {
                        if (current_str != "") {
                            vt_panic("tokenize_line error!");
                        }

                        state = "STRING";
                        current_str.push_back('"');
                        continue;
                    }
                    if ( cur == '\'' ) {
                        if (current_str != "") {
                            vt_panic("tokenize_line error!");
                        }

                        state = "STRING";
                        current_str.push_back('\'');
                        continue;
                    }

                    if ( cur == ';' ) {
                        if (current_str != "") {
                            out.push_back(current_str);
                        }
                        return;
                    }

                    current_str.push_back(cur);
                    continue;
                }
                if ( state == "STRING" ) {
                    if ( cur == '"' && current_str.at(0) == '"') {
                        current_str.push_back('"');
                        out.push_back(current_str);
                        current_str = "";
                        state = "SYMBOL";
                        continue;
                    }
                    if ( cur == '\'' && current_str.at(0) == '\'') {
                        current_str.push_back('\'');
                        out.push_back(current_str);
                        current_str = "";
                        state = "SYMBOL";
                        continue;
                    }
                    current_str.push_back(cur);
                }
            }
            if ( state == "STRING" ) {
                vt_panic("tokenize_line error, string must end in one line!");
            }
            if (current_str != "") {
                out.push_back(current_str);
            }
        }

        static bool is_valid_name(std::string const &str) {
            if ( str == "true" || str == "false" || str == "null" || str == "@" || str == "!" || str == "!!" || str == "jnz" || str =="jz" ) {
                return false;
            }
            if (str.find_first_not_of("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") == std::string::npos) {
                if ( str.find_first_of("0123456789") != 0 ) {
                    return true;
                }
            }
            return false;
        }
    };

    // 0. removed comments & tokenize
    std::vector<std::string> tokens;
    std::istringstream code_stream(txt);
    std::string line;
    while (std::getline(code_stream, line)) {
        _::tokenize_line(line,  tokens);
    }

    // 1. pre-processing code loop
    {
        bool block_comment = false;
        std::vector<std::string> new_tokens;

        // removed comments
        for (size_t i = 0; i < tokens.size(); i++) {
            auto token = tokens[i];
            if ( token == "/*" ) {
                if ( block_comment == true ) {
                    vt_panic("Find /* block comment nesting !");
                }
                block_comment = true;
                continue;
            }
            if ( token == "*/" ) {
                if ( block_comment == false ) {
                    vt_panic("Find */ block comment without /* begin !");
                }
                block_comment = false;
                continue;
            }
            if ( block_comment == true) {
                continue;
            }

            new_tokens.push_back(token);
        }
        tokens = new_tokens;
        new_tokens.clear();

        // extending loop
        bool in_loop = false;
        std::vector<std::string> loop_block;
        std::optional<int> loop_begin;
        std::optional<int> loop_end;
        for (size_t i = 0; i < tokens.size(); i++) {
            auto token = tokens[i];
            if ( token == "%for" ) {
                if ( in_loop == true ) {
                    vt_panic("Find loop block nesting !");
                }
                in_loop = true;
                continue;
            }
            if ( token == "%endf" ) {
                if ( in_loop == false ) {
                    vt_panic("Find %endf  without %for !");
                }
                if ( !loop_begin.has_value() || !loop_end.has_value() ) {
                    vt_panic("Can't find begin and end !");
                }
                if ( loop_end > loop_begin ) {
                    for (int l = loop_begin.value(); l <= loop_end.value(); l++) {
                        std::stringstream ss;
                        ss << l;
                        std::string dst = ss.str();
                        for (size_t j = 0; j < loop_block.size(); j++) {
                            std::string new_str = std::regex_replace( loop_block[j], std::regex("%%"), dst);
                            new_tokens.push_back(new_str);
                        }
                    }
                } else {
                   for (int l = loop_begin.value(); l >= loop_end.value(); l--) {
                        std::stringstream ss;
                        ss << l;
                        std::string dst = ss.str();
                        for (size_t j = 0; j < loop_block.size(); j++) {
                            std::string new_str = std::regex_replace( loop_block[j], std::regex("%%"), dst);
                            new_tokens.push_back(new_str);
                        }
                   }
                }
                loop_block.clear();
                loop_begin.reset();
                loop_end.reset();
                in_loop = false;
                continue;
            }
            if ( in_loop == true ) {
                double num;
                if ( !loop_begin.has_value() ) {
                    if ( !_::parse_number(token, num) ) {
                        vt_panic("can't find loop begin number");
                    }
                    loop_begin = (int)num;
                    continue;
                }
                if ( !loop_end.has_value() ) {
                    if ( !_::parse_number(token, num) ) {
                        vt_panic("can't find loop begin number");
                    }

                    loop_end = (int)num;
                    continue;
                }
                loop_block.push_back(token);
                continue;
            }
            new_tokens.push_back(token);
        }
        tokens = new_tokens;
    }

    UserWord main_code;
    std::optional<UserWord> user_code;
    std::optional<size_t> list_count;
    list_count.reset();

    // 2. processing main code
    for (size_t i = 0; i < tokens.size(); i++) {
        auto token = tokens[i];
        // first pass, processing command primitive
        if ( token == "%def" ) {
            if ( user_code.has_value() ) {
                vt_panic("Can't define a new user word inside another user word!");
            }
            if ( list_count.has_value() ) {
                vt_panic("Can't define a new user word inside a list macro!");
            }
            user_code = UserWord();

            i = i + 1;
            if ( i >= tokens.size() ) {
                vt_panic("Can't find #end for #def!");
            }
            token = tokens[i];
            if ( _::is_valid_name(token) ) {
                if ( user_words_.find(token) == user_words_.end() ) {
                    if ( native_words_.find(token) == native_words_.end() ) {
                        user_code.value().push_back( WordCode::new_string( token) );
                        continue;
                    }
                }
            }
            vt_panic("Can't a valid name for #def macro!");

        } else if ( token == "[" ) {
            list_count = 0;
            continue;
        } else if ( token == "%end" ) {
            if ( list_count.has_value() ) {
                vt_panic("Can't ending a word or in a list macro.");
            }

            if ( user_code.has_value() ) {
                auto w = user_code.value();
                if ( w.size() < 1 ) {
                    vt_panic("define macro error, must including a word name!");
                }
                vt_assert( w[0].type_ == WordCode::String, "first of define macro, must be a name!");

                auto name = w[0].str_;
                w.erase(w.begin());
                user_words_[name] = w;

                user_code.reset();
                continue;
            }

            vt_panic("Find #end without #def");
        } else if ( token == "]" ) {
            if ( !list_count.has_value() ) {
                vt_panic("']' list macro appears without begin '['!");
            }

            UserWord* target = &main_code;
            if ( user_code.has_value() ) {
                target = &user_code.value();
            }

            double lcount = list_count.value();
            target->push_back( WordCode::new_number( lcount ) );

            list_count.reset();
            continue;
        }
        // second pass code to number, string,builtin, native, or user
        WordCode newCode;

        if ( token == "ture" ) {
            newCode = WordCode::new_number( 1.0 );
        } else if ( token == "false") {
            newCode = WordCode::new_number( 0.0 );
        } else if ( token == "null" ) {
            newCode = WordCode::new_string( "" );
        } else if ( token == "@" ||
                    token == "!"  ||
                    token == "!!" ||
                    token == "jnz"||
                    token == "jz" ) {
            newCode = WordCode::new_builtin( token );
        } else if ( token[0] == '"' || token[0] == '\'' || token[0] == '$' ) {
            if ( token[0] == '"' || token[0] == '\'' ) {
                vt_assert( token.size() >= 2, " string must begin \" ' or $");
                vt_assert( token[0] == token.back() , " string length wrong");
                token = token.substr(1, token.size() - 2);
            }
            newCode = WordCode::new_string( token );
        } else if ( _::parse_number(token, newCode.num_) ) {
            newCode = WordCode::new_number( newCode.num_ );
        } else if ( native_words_.find( token ) != native_words_.end() ) {
            newCode = WordCode::new_native( token );
        } else if ( user_words_.find( token ) != user_words_.end() ) {
            newCode = WordCode::new_user( token );
        } else {
            std::cout << token << std::endl;
            vt_panic("Find an invalid symbol is not string, number, builtin, user or native!");
        }

        UserWord* target = &main_code;
        if ( user_code.has_value() ) {
            target = &user_code.value();
        }
        if ( list_count.has_value() ) {
            list_count = list_count.value() + 1;
        }

        target->push_back(newCode);
    }

    if (list_count.has_value()) {
        vt_panic("List macro without ']' ending!");
    }
    if (user_code.has_value()) {
        vt_panic("Define macro without #end ending!");
    }

    return main_code;
}

namespace builtin {
    struct BuiltinGet : public BuiltinOperator {
        int run(Enviroment* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            auto value = hash.find(name);
            stack.push( Hash::Item2Cell(value) );
            return 1;
        }
    };

    struct BuiltinSet : public BuiltinOperator {
        int run(Enviroment* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            Cell cell = stack.pop();
            hash.set(name, Hash::Cell2Item(cell));
            return 1;
        }
    };

    struct BuiltinRemove : public BuiltinOperator {
        BuiltinRemove() {
        }
        int run(Enviroment* env) override {
            auto& hash = env->hash();
            auto& stack = env->stack();

            auto name = stack.pop_string();
            hash.drop(name);
            return 1;
        }
    };

    struct BuiltinJNZ : public BuiltinOperator {
        BuiltinJNZ() {
        }
        int run(Enviroment* env) override {
            auto& stack = env->stack();

            int steps = stack.pop_number();
            int cond = stack.pop_number();
            if ( cond ) {
                return steps;
            }
            return 1;
        }
    };

    struct BuiltinJZ : public BuiltinOperator {
        BuiltinJZ() {
        }
        int run(Enviroment* env) override {
            auto& stack = env->stack();

            int steps = stack.pop_number();
            int cond = stack.pop_number();
            if ( !cond ) {
                return steps;
            }
            return 1;
        }
    };
}


void Enviroment::linking(DaG& dag, UserWord& word) {
    auto& builtins_ = dag.builtins_;
    auto& natives_ = dag.natives_;
    auto& binary_ = dag.binary_;

    for(size_t i = 0; i < word.size(); i++) {
        auto code = word[i];
        switch( code.type_ ) {
            case WordCode::Number :
                binary_.push_back( WordByte( code.num_ ) );
                break;

            case WordCode::String :
                binary_.push_back( WordByte( code.str_  ) );
                break;

            case WordCode::Builtin :
                {
                    BuiltinOperator* op = nullptr;
                    if ( code.str_ == "@" ) {
                        op = new builtin::BuiltinGet();
                    } else if ( code.str_ == "!" ) {
                        op = new builtin::BuiltinSet();
                    } else if ( code.str_ == "!!" ) {
                        op = new builtin::BuiltinRemove();
                    } else if ( code.str_ == "jnz" ) {
                        op = new builtin::BuiltinJNZ();
                    } else if ( code.str_ == "jz" ) {
                        op = new builtin::BuiltinJZ();
                    } else {
                        vt_panic("Find an unsupoorted builtin operator!");
                    }
                    size_t idx = builtins_.size();
                    builtins_.push_back(op);
                    binary_.push_back( WordByte( WordByte::BuiltinOperator, idx) );
                }
                break;

            case WordCode::Native :
                binary_.push_back( WordByte(WordByte::Native, natives_.size() ));
                natives_.push_back( create_native(code.str_));
                break;

            case WordCode::User :
                UserWord& new_word = get_user( code.str_ );
                linking(dag, new_word);
                break;
        }
    }
}

void Enviroment::run_(DaG* dag) {
    auto& binary_ = dag->binary_;
    auto& builtins_ = dag->builtins_;
    auto& natives_ = dag->natives_;

    for ( size_t pc = 0; pc < binary_.size(); ) {
        auto& byte = binary_[pc];
        int steps = 1;
        switch( byte.type_ ) {
            case WordByte::Number:
                stack_->push_number( byte.num_ );
                break;

            case WordByte::String:
                stack_->push_string( byte.str_ );
                break;

            case WordByte::BuiltinOperator:
                steps = builtins_[ byte.idx_ ]->run( this );
                break;

            case WordByte::Native:
                natives_[ byte.idx_ ]->run( *stack_ );
                break;

            default:
                vt_panic("Runing binary error can't bere!");
                break;
        }
        vt_assert( steps > 0, " DAG don't support loop ");
        pc += steps;
    }
}

namespace base {
    struct Exit : public NativeWord {
        void run(Stack& stack) override {
            exit(0);
        }
        NWORD_CREATOR_DEFINE_LR(Exit)
    };

    struct Dump : public NativeWord {
        void run(Stack& stack) override {
            std::cout << stack << std::endl;
        }
        NWORD_CREATOR_DEFINE_LR(Dump)
    };

    struct Echo : public NativeWord {
        void run(Stack& stack) override {
            Cell cell = stack.pop();
            std::cout << cell << std::endl;
        }
        NWORD_CREATOR_DEFINE_LR(Echo)
    };

    struct Drop : public NativeWord {
        void run(Stack& stack) override {
            stack.drop();
        }
        NWORD_CREATOR_DEFINE_LR(Drop)
    };

    struct Dup : public NativeWord {
        void run(Stack& stack) override {
            stack.dup();
        }
        NWORD_CREATOR_DEFINE_LR(Dup)
    };

    struct Dup2 : public NativeWord {
        void run(Stack& stack) override {
            stack.dup2();
        }
        NWORD_CREATOR_DEFINE_LR(Dup2)
    };

    struct Swap : public NativeWord {
        void run(Stack& stack) override {
            stack.swap();
        }
        NWORD_CREATOR_DEFINE_LR(Swap)
    };

    struct Rot : public NativeWord {
        void run(Stack& stack) override {
            stack.rot();
        }
        NWORD_CREATOR_DEFINE_LR(Rot)
    };

    struct Rev : public NativeWord {
        void run(Stack& stack) override {
            int n = stack.pop_number();
            stack.rev(n);
        }
        NWORD_CREATOR_DEFINE_LR(Rev)
    };

    struct Add : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b + a );
        }
        NWORD_CREATOR_DEFINE_LR(Add)
    };

    struct Sub : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b - a );
        }
        NWORD_CREATOR_DEFINE_LR(Sub)
    };

    struct Mul : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b * a );
        }
        NWORD_CREATOR_DEFINE_LR(Mul)
    };

    struct Div : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( b / a );
        }
        NWORD_CREATOR_DEFINE_LR(Div)
    };

    struct IntDiv : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( (int)(b / a) );
        }
        NWORD_CREATOR_DEFINE_LR(IntDiv)
    };

    struct Equals : public NativeWord {
        void run(Stack& stack) override {
            double a = stack.pop_number();
            double b = stack.pop_number();
            stack.push_number( a == b);
        }
        NWORD_CREATOR_DEFINE_LR(Equals)
    };

    struct Combin : public NativeWord {
        void run(Stack& stack) override {
            auto a = stack.pop_string();
            auto b = stack.pop_string();
            std::string c = b + a;
            stack.push_string( c );
        }
        NWORD_CREATOR_DEFINE_LR(Combin)
    };


}

Enviroment::Enviroment(ComputingContext* ctx)
    : stack_(std::make_unique<Stack>()), hash_(std::make_unique<Hash>()), ctx_(ctx) {
    load_base_words();
}

Enviroment::~Enviroment() = default;
void Enviroment::load_base_words() {
    // base words
    insert_native_word("drop", base::Drop::creator );
    insert_native_word("dup", base::Dup::creator );
    insert_native_word("dup2", base::Dup2::creator );
    insert_native_word("swap", base::Swap::creator );
    insert_native_word("rot", base::Rot::creator );
    insert_native_word("rev", base::Rev::creator );
    insert_native_word("?", base::Echo::creator );
    insert_native_word("??", base::Dump::creator );
    insert_native_word("^", base::Exit::creator );

    insert_native_word("+", base::Add::creator );
    insert_native_word("-", base::Sub::creator );
    insert_native_word("*", base::Mul::creator );
    insert_native_word("/", base::Div::creator );
    insert_native_word("//", base::IntDiv::creator );
    insert_native_word("==", base::Equals::creator );

    insert_native_word("|", base::Combin::creator );
}


}
