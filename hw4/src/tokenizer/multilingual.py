import unicodedata

from tokenizer.bpe import UnicodeBPETokenizer


def visualize_bytes(b: bytes):
    return "".join(
        ch
        for ch in b.decode("utf-8", errors="replace")
        if unicodedata.category(ch)[0] != "C"
    )


english_tokenizer = UnicodeBPETokenizer.from_config("data/english-tokenizer.json")
english_str = "Mark Gwyn (1962/1963 – August 2024) was an American law enforcement officer. He was the director of the Tennessee Bureau of Investigation (TBI). He was eighth director in the agency's history and the first African American to serve in this capacity."
thai_str = "มาร์ก กวิน (1962/1963 – สิงหาคม 2024) เป็นเจ้าหน้าที่บังคับใช้กฎหมายชาวอเมริกัน เขาเป็นผู้อำนวยการสำนักงานสืบสวนสอบสวนแห่งรัฐเทนเนสซี (TBI) เขาเป็นผู้อำนวยการคนที่แปดในประวัติศาสตร์ของหน่วยงานและเป็นชาวแอฟริกันอเมริกันคนแรกที่รับหน้าที่นี้"

print("document length in English", len(english_tokenizer.encode(english_str)))
print("document length in Thai", len(english_tokenizer.encode(thai_str)))
