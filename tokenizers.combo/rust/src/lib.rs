pub mod tiktoken;
pub mod tiktoken_wrapper;

use load_image::*;

pub struct ImageObject {
    width:  usize,
    height: usize,
    rgb: Vec<u8>
}

#[no_mangle]
extern "C" fn imgobj_load(_file: *const u8, _len: usize) -> *mut ImageObject {
    unsafe {
        let path = std::str::from_utf8(std::slice::from_raw_parts(_file, _len)).unwrap();
        let _img = load_image::load_path(path).unwrap();
       	let rgb = match _img.bitmap {
            ImageData::RGB8(ref bitmap) => {
                let mut d : Vec<u8> = vec![];
                for bgr in bitmap.iter() {
                    d.push(bgr.b);
                    d.push(bgr.g);
                    d.push(bgr.r);
                };
                d
            },
            _ => panic!(),
        };
        let image = Box::new(ImageObject {
            width: _img.width,
            height: _img.height,
            rgb: rgb
        });
        return Box::into_raw(image);
    }
}

#[no_mangle]
extern "C" fn imgobj_width(handle: *mut ImageObject ) -> usize {
    unsafe {
        (*handle).width
    }
}

#[no_mangle]
extern "C" fn imgobj_height(handle: *mut ImageObject ) -> usize {
    unsafe {
        (*handle).height
    }
}

#[no_mangle]
extern "C" fn imgobj_rgb_plane(handle: *mut ImageObject, rgb: *mut u8) {
    unsafe {
        let w = (*handle).width;
        let h = (*handle).height;
        let plane = w * h;
        for i in 0..h {
            for j in 0..w {
                let ii : usize = j + i * w;
                *rgb.add(ii) = (*handle).rgb[ii*3 + 0];
                *rgb.add(plane + ii) = (*handle).rgb[ii*3 + 1];
                *rgb.add(plane * 2 + ii) = (*handle).rgb[ii*3 + 2];
            }
        }
    }
}

#[no_mangle]
extern "C" fn imgobj_free(handle: *mut ImageObject) {
    unsafe {
        drop(Box::from_raw(handle));
    }
}

