use crate::tiktoken::CoreBPE;

pub struct TiktokenWrapper {
    tokenizer: CoreBPE,
    encode_ids: Vec<u32>,
    decode_str: String,
}

impl TiktokenWrapper {
    pub fn new( enc_lines: &str, spec_lines: &str, reg: &str) -> Self {
        let bpe = CoreBPE::create_from(enc_lines, spec_lines, reg);
        let ids:Vec<u32> = Vec::new();
        let dstr:String = String::new();

        TiktokenWrapper {
            tokenizer:  bpe,
            encode_ids: ids,
            decode_str: dstr
        }
    }

    pub fn encode(&mut self, input: &str) {
        let ids: Vec<usize> = self.tokenizer.encode(input);
        self.encode_ids = ids.iter().map(|us| *us as u32).collect();
    }

    pub fn encode_ordinary(&mut self, input: &str) {
        let ids: Vec<usize> = self.tokenizer.encode_ordinary(input);
        self.encode_ids = ids.iter().map(|us| *us as u32).collect();
    }

    pub fn decode(&mut self, ids: &[u32]) {
        let ids_: Vec<usize> = ids.iter().map(|ui| *ui as usize).collect();
        self.decode_str = self.tokenizer.decode(&ids_);
    }
}

#[no_mangle]
extern "C" fn tiktoken_new_from_str(enc_cstr: *const u8, enc_len: usize,
                                    spec_cstr: *const u8, spec_len: usize,
                                    reg_cstr: *const u8, reg_len: usize) -> *mut TiktokenWrapper {
    unsafe {
        let enc_lines = std::str::from_utf8(std::slice::from_raw_parts(enc_cstr, enc_len)).unwrap();
        let spec_lines = std::str::from_utf8(std::slice::from_raw_parts(spec_cstr, spec_len)).unwrap();
        let reg = std::str::from_utf8(std::slice::from_raw_parts(reg_cstr, reg_len)).unwrap();

        return Box::into_raw(Box::new(TiktokenWrapper::new(&enc_lines, &spec_lines, &reg)));
    }
}

#[no_mangle]
extern "C" fn tiktoken_encode(
    handle: *mut TiktokenWrapper,
    input_cstr: *const u8,
    len: usize,
    with_special_tokens: i32,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        if with_special_tokens != 0 {
            (*handle).encode(input_data);
        } else {
            (*handle).encode_ordinary(input_data);
        }
    }
}

#[no_mangle]
extern "C" fn tiktoken_get_encode_ids(
    handle: *mut TiktokenWrapper,
    out_data: *mut *mut u32,
    out_len: *mut usize,
) {
    unsafe {
        *out_data = (*handle).encode_ids.as_mut_ptr();
        *out_len = (*handle).encode_ids.len()
    }
}

#[no_mangle]
extern "C" fn tiktoken_decode(
    handle: *mut TiktokenWrapper,
    input_ids: *const u32,
    len: usize,
) {
    unsafe {
        let input_data = std::slice::from_raw_parts(input_ids, len);
        (*handle).decode(input_data);
    }
}

#[no_mangle]
extern "C" fn tiktoken_get_decode_str(
    handle: *mut TiktokenWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (*handle).decode_str.len();
    }
}

#[no_mangle]
extern "C" fn tiktoken_free(wrapper: *mut TiktokenWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}
