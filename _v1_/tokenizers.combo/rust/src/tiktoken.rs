use std::collections::HashSet;
use std::collections::HashMap;
use std::str;
use std::thread;
use fancy_regex::Regex;
use base64;

fn _byte_pair_merge<T>(
    piece: &[u8],
    ranks: &HashMap<Vec<u8>, usize>,
    f: impl Fn(std::ops::Range<usize>) -> T,
) -> Vec<T> {
    // This is a vector of (start, rank).
    // The rank is of the byte pair starting at position start.
    // The rank of the last item in the vector is not a valid value.
    let mut parts: Vec<(usize, usize)> = (0..piece.len() + 1).map(|i| (i, usize::MAX)).collect();

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, usize)>, start_idx: usize, skip: usize| {
            if (start_idx + skip + 2) < parts.len() {
                ranks
                    .get(&piece[parts[start_idx].0..parts[start_idx + skip + 2].0])
                    .copied()
            } else {
                None
            }
        }
    };

    // We look up the ranks once in the beginning and iteratively update
    // them during each merge, which reduces the number of rank lookups.
    for i in 0..parts.len() - 2 {
        match get_rank(&parts, i, 0) {
            Some(rank) => {
                // usize::MAX is a sentinel value and cannot be a valid rank
                debug_assert!(rank != usize::MAX);
                parts[i].1 = rank;
            }
            None => {
                continue;
            }
        };
    }

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // It is important to consider that n is often small (<100), and as such
    // the cache-locality benefits outweigh the algorithmic complexity downsides
    // of the `parts` vector data structure above.

    // Note that we hash bytes, not token pairs. As long as we train BPE the way we
    // currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    loop {
        if parts.len() == 1 {
            break;
        }

        // usize::MAX is a sentinel rank value allowing us to
        // take the min more quickly
        let mut min_rank: (usize, usize) = (usize::MAX, 0);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }

        if min_rank.0 != usize::MAX {
            let i = min_rank.1;

            // NOTE: We are about to remove parts[i + 1]. We do not do it
            // yet because there are cache-locality benefits to updating
            // parts[i] and parts[i-1] before removing, which could thrash
            // the cache. Thus, we update the rank calculation by skipping over
            // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
            parts[i].1 = get_rank(&parts, i, 1).unwrap_or(usize::MAX);
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1, 1).unwrap_or(usize::MAX);
            }

            parts.remove(i + 1);
        } else {
            break;
        }
    }
    let mut out: Vec<T> = Vec::with_capacity(parts.len() - 1);
    for i in 0..parts.len() - 1 {
        out.push(f(parts[i].0..parts[i + 1].0));
    }
    out
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<usize> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(piece, ranks, |p| ranks[&piece[p.start..p.end]])
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<&'a [u8]> {
    if piece.len() == 1 {
        return vec![piece];
    }
    _byte_pair_merge(piece, ranks, |p| &piece[p.start..p.end])
}

// Various performance notes:
//
// Regex
// =====
// Most of the time is spent in regex. The easiest way to speed this up is by using less fancy
// regex features. For instance, using a regex parse-able by `regex` crate is 3x faster than
// the usual regex we use.
//
// However, given that we're using a regex parse-able by `regex`, there isn't much difference
// between using the `regex` crate and using the `fancy_regex` crate.
//
// There is an important interaction between threading, `regex` and `fancy_regex`.
// When using `fancy_regex`, we hit `regex.find_at`. It turns out that this causes contention on
// some mutable scratch space inside of `regex`. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `find_iter` has a different code path.
// Related: https://github.com/rust-lang/regex/blob/master/PERFORMANCE.md
// Anyway, the way we get around this is with having a (mostly) thread local clone of the regex for
// each thread.
//
// Threading
// =========
// I tried using `rayon`. It wasn't really faster than using Python threads and releasing the GIL.
// So goodbye `rayon`! Let thread count etc be in control of our Python users.
//
// Caching
// =======
// The reference tokeniser has an lru cache over the equivalent of `byte_pair_encode`.
// Originally, we had one too! Without it, we were only vaguely faster than Python.
// I used an RWLock to protect the cache. This didn't seem to hurt single threaded performance
// noticeably, but it did affect multi-threaded performance. Weirdly, it seemed to affect
// multi-threaded performance even when I only had readers (maybed I messed something up?).
// Anyway, I realised that we could get rid of the cache, if we treat the set of tokens as a cache!
// These are exactly the set or merges that are likely to be hot. And now we don't have to think
// about interior mutability, memory use, or cloning.
//
// Hashing
// =======
// We use FxHashMap instead of the standard HashMap. This is maybe like a 5-10% win?
// The current implementation ends up doing a lot of hashing of bytes. In theory, this could be made
// to be hashing of two-tuples of ints, which looks like it may also be a couple percent faster.

use std::num::NonZeroU64;
pub struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

const MAX_NUM_THREADS: usize = 8;
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, usize>,
    special_tokens_encoder: HashMap<String, usize>,
    decoder: HashMap<usize, Vec<u8>>,
    special_tokens_decoder: HashMap<usize, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
}

impl CoreBPE {
    fn _get_tl_regex(&self) -> &Regex {
        // See performance notes above for what this is about
        // It's also a little janky, please make a better version of it!
        // However, it's nice that this doesn't leak memory to short-lived threads
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _decode_native(&self, tokens: &[usize]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = self
                .decoder
                .get(token)
                .unwrap_or_else(|| &self.special_tokens_decoder[token]);
            ret.extend(token_bytes);
        }
        ret
    }

    fn _encode_ordinary_native(&self, text: &str) -> Vec<usize> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            if let Some(token) = self.encoder.get(piece) {
                ret.push(*token);
                continue;
            }
            ret.extend(&byte_pair_encode(piece, &self.encoder));
        }
        ret
    }

    fn _encode_native(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<usize>, usize) {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                // Find the next allowed special token, if any
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to _encode_ordinary_native
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        (ret, last_piece_token_len)
    }

    fn _encode_native_all(&self, text: &str) -> (Vec<usize>, usize) {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let next_special = special_regex.find_from_pos(text, start).unwrap();
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to _encode_ordinary_native
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        (ret, last_piece_token_len)
    }


    fn _increase_last_piece_token_len(
        &self,
        tokens: Vec<usize>,
        mut last_piece_token_len: usize,
    ) -> (Vec<usize>, usize) {
        // Unfortunately, the locations where our regex splits can be unstable.
        // For the purposes of determining unstable tokens, unstable regex splitting
        // is only a problem if a split that was present disappears, since this can
        // lead to merging of tokens otherwise thought to be stable.
        // cl100k_base makes our life hard by including the \s*[\r\n]+
        // pattern. This can e.g. cause "\n" + " " to become "\n \n".
        // Here is a quick and dirty fix:
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
            {
                while (last_piece_token_len < tokens.len())
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }
}

impl CoreBPE {
    fn new (
        encoder: HashMap<Vec<u8>, usize>,
        special_tokens_encoder: HashMap<String, usize>,
        pattern: &str,
    ) -> Result<Self, String> {
        let regex = Regex::new(pattern)
            .map_err(|e| e.to_string())?;

        let special_regex = {
            let _parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|"))
                .map_err(|e| e.to_string())?
        };

        let decoder: HashMap<usize, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<usize, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        Ok(CoreBPE {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(),
            special_regex_tls: (0..MAX_NUM_THREADS)
                .map(|_| special_regex.clone())
                .collect()
        })
    }

    pub fn create_from(enc_lines: &str, spec_lines: &str, reg: &str) -> CoreBPE {
        let mut enc:HashMap<Vec<u8>, usize> = HashMap::new();
        {
            let lines : Vec<&str> = enc_lines.split('\n').collect();

            for line in lines {
                if  line == "" {
                    break;
                }
                let (piece_s, id_s) = {
                    let words : Vec<&str> = line.split(' ').collect();
                    (words[0], words[1])
                };

                let id = id_s.parse::<usize>().unwrap();
                let piece = base64::decode( piece_s).unwrap();

                enc.insert(piece, id);
            }
        }

        let mut spec: HashMap<String, usize> = HashMap::new();
        {
            let lines : Vec<&str> = spec_lines.split('\n').collect();

            for line in lines {
                if  line == "" {
                    break;
                }

                let id = enc.len() + spec.len();
                let piece = line.to_string();

                spec.insert(piece, id);
            }
        }

        CoreBPE::new(enc, spec, reg).unwrap()
    }

    // ====================
    // Encoding
    // ====================
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self._encode_native_all(text).0
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        self._encode_ordinary_native(text)
    }

    pub fn encode_with_allowed(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<usize> {
        self._encode_native(text, &allowed_special).0
    }

    // ====================
    // Decoding
    // ====================
    pub fn decode(&self, tokens: &[usize]) -> String  {
        let bytes = self._decode_native(&tokens);
        let s = str::from_utf8(&bytes);
        if s.is_ok() {
            return s.unwrap().to_string();
        }
        return "".to_string();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn it_works() {
        let file_path = "/root/workspace/br/models/qwen-7b/qwen.tiktoken";
        let enc_contents = fs::read_to_string(file_path).unwrap();
        let reg = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
        let specs = "<unk>\n<bos>\n<eos>\n";
        let tok = crate::tiktoken::CoreBPE::create_from(&enc_contents, specs, reg);
        let ret = tok.encode("测试效果 hello world in the blue sky.");
        for ri in &ret {
            println!("{}", *ri);
        }
        let s = tok.decode(&ret);
        println!("{}", s);
    }
}
