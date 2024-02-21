pub mod tiktoken;
pub mod tiktoken_wrapper;

use image::imageops::FilterType;

pub struct ImageObject {
    img : image::DynamicImage
}


#[no_mangle]
extern "C" fn imgobj_load(_file: *const u8, _len: usize) -> *mut ImageObject {
    unsafe {
        let path = std::str::from_utf8(std::slice::from_raw_parts(_file, _len)).unwrap();
        let origin = image::open(path).unwrap();
        let image = Box::new(ImageObject {
            img : origin
        });

        return Box::into_raw(image);
    }
}

#[no_mangle]
extern "C" fn imgobj_free(handle: *mut ImageObject) {
    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
extern "C" fn imgobj_width(handle: *mut ImageObject ) -> u32 {
    let img : &ImageObject = unsafe {
        &*handle
    };
    return img.img.width();
}

#[no_mangle]
extern "C" fn imgobj_height(handle: *mut ImageObject ) -> u32 {
    let img : &ImageObject = unsafe {
        &*handle
    };
    return img.img.height();
}

#[no_mangle]
extern "C" fn imgobj_resize(handle: *mut ImageObject, width: u32, height: u32) {
    let img : &mut ImageObject = unsafe {
        &mut *handle
    };

    let scaled = img.img.resize_exact(width, height, FilterType::CatmullRom);
    img.img = scaled;
}


#[no_mangle]
extern "C" fn imgobj_rgb_plane(handle: *mut ImageObject, plane: *mut u8) {
    let img : &ImageObject = unsafe {
        & *handle
    };

    let width = img.img.width();
    let height = img.img.height();
    let psize:usize = (width * height) as usize;
    let rgb = img.img.to_rgb8();

    for h in 0..height {
        for w in 0..width {
            let i:usize = (h * width + w) as usize;
            unsafe {
                *plane.add( i ) = rgb.get_pixel(w, h)[0];
                *plane.add( i + psize) = rgb.get_pixel(w, h)[1];
                *plane.add( i + psize*2) = rgb.get_pixel(w, h)[2];
            }
        }
    }
}


