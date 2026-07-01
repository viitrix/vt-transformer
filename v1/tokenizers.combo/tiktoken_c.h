/*!
 *  Copyright (c) 2023 by Contributors
 * \file tiktoken_c.h
 * \brief C binding to tokenizers rust library
 */
#ifndef TIKTOKEN_C_H_
#define TIKTOKEN_C_H_

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TiktokenHandle;

TiktokenHandle tiktoken_new_from_str(const char* enc, size_t  enc_len,
                                     const char* spec, size_t spec_len,
                                     const char* reg, size_t reg_len);

void tiktoken_encode(TiktokenHandle handle, const char* data, size_t len, int add_special_token);

void tiktoken_decode(TiktokenHandle handle, const uint32_t* data, size_t len,
                       int skip_special_token);

void tiktoken_get_decode_str(TiktokenHandle handle, const char** data, size_t* len);

void tiktoken_get_encode_ids(TiktokenHandle handle, const uint32_t** id_data, size_t* len);

void tiktoken_free(TiktokenHandle handle);

#ifdef __cplusplus
}
#endif
#endif  // TIKTOKEN_C_H_
