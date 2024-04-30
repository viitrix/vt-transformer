import json
import base64
import array

def bytes_to_unicode():
    """                                                                                                                                Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))


with open("./tokenizer.json") as f:
    vocabs = json.load(f)["model"]["vocab"];

tiks = [-1] * len(vocabs);
for key in vocabs:
    v = vocabs[key];
    tiks[v] = key;

bd = bytes_to_unicode();


of = open("./llama3.tiktoken", "w");

for i in range(0, len(tiks)):
    s = b'';
    v = tiks[i];
    for j in range(0, len(v)):
        s = s + bd[ v[j] ].to_bytes();
    r = base64.b64encode(s).decode('ascii');
    r = r + " " + str(i) + "\n";
    of.write(r);

of.close();


