// vt_test.cpp — Tests for the vt DSL (vt.hpp / vt.cpp)
//
// Only vt.hpp is included; vt.cpp is linked as a separate translation unit.
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I.. vt_test.cpp ../core/vt.cpp -o /tmp/vt_test
// Run:
//   /tmp/vt_test

#include <core/vt.hpp>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << (msg) << std::endl;                                \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

#define CHECK_EQ(a, b, msg)                                                 \
    do {                                                                    \
        auto _a = (a);                                                      \
        auto _b = (b);                                                      \
        if (!(_a == _b)) {                                                  \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << (msg) << " (got " << _a << ", want " << _b << ")"  \
                      << std::endl;                                         \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

using vt::Enviroment;
using vt::ComputingContext;

// ============================================================================
// Number literals & arithmetic
// ============================================================================

void test_push_integer() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("42");
    CHECK_EQ(env.stack().size(), 1u, "stack size after push");
    CHECK_EQ(env.stack().pop_number(), 42.0, "integer literal");
}

void test_push_negative() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("-5");
    CHECK_EQ(env.stack().pop_number(), -5.0, "negative integer");
}

void test_push_float() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("3.14");
    CHECK_EQ(env.stack().pop_number(), 3.14, "float literal");
}

void test_add() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("3 4 +");
    CHECK_EQ(env.stack().pop_number(), 7.0, "3 + 4");
}

void test_sub() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("10 3 -");
    CHECK_EQ(env.stack().pop_number(), 7.0, "10 - 3");
}

void test_mul() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 6 *");
    CHECK_EQ(env.stack().pop_number(), 30.0, "5 * 6");
}

void test_div() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("20 4 /");
    CHECK_EQ(env.stack().pop_number(), 5.0, "20 / 4");
}

void test_intdiv() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("17 5 //");
    CHECK_EQ(env.stack().pop_number(), 3.0, "17 // 5");
}

void test_arithmetic_precedence_postfix() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // (1 + 2) * 3 = 9 in postfix: 1 2 + 3 *
    env.execute("1 2 + 3 *");
    CHECK_EQ(env.stack().pop_number(), 9.0, "(1+2)*3");
}

// ============================================================================
// Equality
// ============================================================================

void test_equals_true() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 5 ==");
    CHECK_EQ(env.stack().pop_number(), 1.0, "5 == 5");
}

void test_equals_false() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 6 ==");
    CHECK_EQ(env.stack().pop_number(), 0.0, "5 == 6");
}

// ============================================================================
// Stack ops
// ============================================================================

void test_dup() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 dup");
    CHECK_EQ(env.stack().size(), 2u, "size after dup");
    CHECK_EQ(env.stack().pop_number(), 5.0, "top");
    CHECK_EQ(env.stack().pop_number(), 5.0, "next");
}

void test_drop() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 3 drop");
    CHECK_EQ(env.stack().size(), 1u, "size after drop");
    CHECK_EQ(env.stack().pop_number(), 5.0, "remaining");
}

void test_swap() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("5 3 swap");
    CHECK_EQ(env.stack().pop_number(), 5.0, "top after swap");
    CHECK_EQ(env.stack().pop_number(), 3.0, "next after swap");
}

void test_rot() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Stack [1 2 3] (top=3). rot → [2 3 1] (top=1).
    env.execute("1 2 3 rot");
    CHECK_EQ(env.stack().pop_number(), 1.0, "top after rot");
    CHECK_EQ(env.stack().pop_number(), 3.0, "mid after rot");
    CHECK_EQ(env.stack().pop_number(), 2.0, "bottom after rot");
}

// ============================================================================
// Strings
// ============================================================================

void test_string_double_quoted() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("\"hello\"");
    CHECK_EQ(env.stack().pop_string(), std::string("hello"), "double-quoted");
}

void test_string_single_quoted() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("'hello'");
    CHECK_EQ(env.stack().pop_string(), std::string("hello"), "single-quoted");
}

void test_string_concat() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("\"hello \" \"world\" |");
    CHECK_EQ(env.stack().pop_string(), std::string("hello world"), "concat with |");
}

// ============================================================================
// Hash (@, !, !!)
// ============================================================================

void test_hash_set_get() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // "alice" "name" !  →  name = alice
    // "name" @          →  push alice
    env.execute("\"alice\" \"name\" ! \"name\" @");
    CHECK_EQ(env.stack().size(), 1u, "stack has 1 after roundtrip");
    CHECK_EQ(env.stack().pop_string(), std::string("alice"), "hash roundtrip");
}

void test_hash_set_number() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // 42 "answer" !  →  answer = 42
    env.execute("42 \"answer\" ! \"answer\" @");
    CHECK_EQ(env.stack().pop_number(), 42.0, "hash number roundtrip");
}

void test_hash_remove() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // set, then remove; final stack should be empty
    env.execute("\"v\" \"k\" ! \"k\" !!");
    CHECK_EQ(env.stack().size(), 0u, "stack empty after remove");
}

// ============================================================================
// List macro [ ... ]
// ============================================================================

void test_list_macro() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // [ 1 2 3 ] pushes 1, 2, 3, then the count 3
    env.execute("[ 1 2 3 ]");
    CHECK_EQ(env.stack().size(), 4u, "3 values + count");
    CHECK_EQ(env.stack().pop_number(), 3.0, "count on top");
}

// ============================================================================
// User words (%def / %end)
// ============================================================================

void test_user_word_simple() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("%def inc 1 + %end 5 inc");
    CHECK_EQ(env.stack().pop_number(), 6.0, "5 inc = 6");
}

void test_user_word_square() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("%def square dup * %end 4 square");
    CHECK_EQ(env.stack().pop_number(), 16.0, "4 square = 16");
}

void test_user_word_call_twice() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // inc adds 1; calling twice: 10 inc inc → 12
    env.execute("%def inc 1 + %end 10 inc inc");
    CHECK_EQ(env.stack().pop_number(), 12.0, "10 inc inc = 12");
}

// ============================================================================
// %for / %endf loop
// ============================================================================

void test_for_loop_ascending() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Expands to: 1 2 3
    env.execute("%for 1 3 %% %endf");
    CHECK_EQ(env.stack().size(), 3u, "3 iterations");
    CHECK_EQ(env.stack().pop_number(), 3.0, "last (top)");
    CHECK_EQ(env.stack().pop_number(), 2.0, "mid");
    CHECK_EQ(env.stack().pop_number(), 1.0, "first (bottom)");
}

void test_for_loop_descending() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Expands to: 3 2 1
    env.execute("%for 3 1 %% %endf");
    CHECK_EQ(env.stack().size(), 3u, "3 iterations");
    CHECK_EQ(env.stack().pop_number(), 1.0, "last (top, descending)");
}

void test_for_loop_with_body() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Each iteration pushes %% and adds 100; final stack: 101 102 103 (top=103)
    env.execute("%for 1 3 %% 100 + %endf");
    CHECK_EQ(env.stack().size(), 3u, "3 iterations");
    CHECK_EQ(env.stack().pop_number(), 103.0, "1+100, 2+100, 3+100 (top)");
}

// ============================================================================
// Comments
// ============================================================================

void test_line_comment() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("42 ; rest of line is ignored");
    CHECK_EQ(env.stack().pop_number(), 42.0, "number before ; survives");
}

void test_line_comment_multiline() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("1\n; comment line\n2 +");
    CHECK_EQ(env.stack().pop_number(), 3.0, "1 + 2 across comment line");
}

void test_block_comment() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("/* ignored block */ 42");
    CHECK_EQ(env.stack().pop_number(), 42.0, "block comment skipped");
}

void test_block_comment_multiline() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    env.execute("/*\n multi\n line\n */ 42");
    CHECK_EQ(env.stack().pop_number(), 42.0, "multiline block comment");
}

// ============================================================================
// Programs that combine multiple features
// ============================================================================

void test_combined_def_and_loop() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Define square, then apply to 1..3
    env.execute("%def square dup * %end %for 1 3 %% square %endf");
    CHECK_EQ(env.stack().size(), 3u, "3 squares");
    CHECK_EQ(env.stack().pop_number(), 9.0, "3^2");
    CHECK_EQ(env.stack().pop_number(), 4.0, "2^2");
    CHECK_EQ(env.stack().pop_number(), 1.0, "1^2");
}

void test_combined_string_and_hash() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // Build a greeting and store/retrieve it
    env.execute("\"hello\" \" \" \"world\" | | \"greeting\" ! \"greeting\" @");
    CHECK_EQ(env.stack().pop_string(), std::string("hello world"), "stored greeting");
}

void test_sum_via_list_macro() {
    ComputingContext ctx;
    Enviroment env(&ctx);
    // [ 1 2 3 ] pushes 1 2 3 3; the trailing 3 is a count we drop, then sum
    env.execute("[ 1 2 3 ] drop + +");
    CHECK_EQ(env.stack().pop_number(), 6.0, "1+2+3 via list macro");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        // numbers & arithmetic
        test_push_integer,
        test_push_negative,
        test_push_float,
        test_add,
        test_sub,
        test_mul,
        test_div,
        test_intdiv,
        test_arithmetic_precedence_postfix,
        // equality
        test_equals_true,
        test_equals_false,
        // stack ops
        test_dup,
        test_drop,
        test_swap,
        test_rot,
        // strings
        test_string_double_quoted,
        test_string_single_quoted,
        test_string_concat,
        // hash
        test_hash_set_get,
        test_hash_set_number,
        test_hash_remove,
        // list macro
        test_list_macro,
        // user words
        test_user_word_simple,
        test_user_word_square,
        test_user_word_call_twice,
        // for loop
        test_for_loop_ascending,
        test_for_loop_descending,
        test_for_loop_with_body,
        // comments
        test_line_comment,
        test_line_comment_multiline,
        test_block_comment,
        test_block_comment_multiline,
        // combined
        test_combined_def_and_loop,
        test_combined_string_and_hash,
        test_sum_via_list_macro,
    };

    for (TestFn t : tests) {
        t();
    }

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED (" << sizeof(tests)/sizeof(tests[0]) << ")\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)\n";
    return 1;
}
