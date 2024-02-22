#ifndef _TOKENIZER_IMAGE_HPP_
#define _TOKENIZER_IMAGE_HPP_

#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <vt.hpp>

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* ImageHandle;

ImageHandle imgobj_load(const char* file_name, size_t  _len);
size_t imgobj_width(ImageHandle handle);
size_t imgobj_height(ImageHandle handle);
void imgobj_rgb_plane(ImageHandle handle, unsigned char *data);
void imgobj_resize(ImageHandle handle, unsigned int width, unsigned int height);
void imgobj_free(ImageHandle handle);

#ifdef __cplusplus
}
#endif

#endif

