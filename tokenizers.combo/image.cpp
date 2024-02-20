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
void imgobj_free(ImageHandle handle);

#ifdef __cplusplus
}
#endif

namespace vt {
    void load_image_rgb_plane(const std::string& filename, std::vector<uint8_t>& rgb) {
        ImageHandle img = imgobj_load(filename.c_str(), filename.size());
        printf(">>>>>>>>>>>>>>> %p\n", img);
        size_t width = imgobj_width(img);
        size_t height = imgobj_height(img);

        rgb.resize(width * height * 3);

        imgobj_rgb_plane(img, rgb.data());
        imgobj_free(img);
    }

}
