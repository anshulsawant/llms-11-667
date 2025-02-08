import unicodedata

from tokenizer.bpe import UnicodeBPETokenizer


def visualize_bytes(b: bytes):
    return "".join(
        ch
        for ch in b.decode("utf-8", errors="replace")
        if unicodedata.category(ch)[0] != "C"
    )


english_tokenizer = UnicodeBPETokenizer.from_config("data/english-tokenizer.json")
for b in english_tokenizer.vocab[256:]:
    print(f"bytes: {b} <-> token: <{visualize_bytes(b)}>")
