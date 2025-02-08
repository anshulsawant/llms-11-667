import datasets as ds
from transformers import AutoTokenizer
from tokenizer.bpe import ASCIIBPETokenizer, string_to_ascii

from pytest_utils.decorators import max_score


def test_not_injective():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    s1 = "Surely you're joking, Mr. Feynman!"
    s2 = "Surely you're joking,Mr. Feynman! "

    assert s1 != s2 and tokenizer.encode(
        s1, add_special_tokens=False
    ) == tokenizer.encode(s2, add_special_tokens=False)


def test_not_invertible():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    s = ';-)'

    s_recovered = tokenizer.decode(tokenizer.encode(s, add_special_tokens=False))
    assert s != s_recovered


def test_not_preserving_concat():
    # you can try to break "google-bert/bert-base-cased"
    # or another tokenizer you like: https://huggingface.co/models?pipeline_tag=text-generation
    tokenizer_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    a = 'hi th'
    b = 'ere'
    assert tokenizer.encode(a + b, add_special_tokens=False) != tokenizer.encode(
        a, add_special_tokens=False
    ) + tokenizer.encode(b, add_special_tokens=False)


@max_score(5)
def test_encode():
    tokenizer = ASCIIBPETokenizer()
    random_ascii = r"[(A{u s=#M \tF\@`P({xAS%"
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)

    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123123") == [128, 51, 128, 51]
    assert tokenizer.encode("12312312") == [128, 51, 128, 51, 128]

    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123") == [129]
    assert tokenizer.encode("123123") == [129, 129]
    assert tokenizer.encode("12312312") == [129, 129, 128]


@max_score(5)
def test_decode():
    tokenizer = ASCIIBPETokenizer()
    assert (
        tokenizer.decode([97, 102, 101, 106, 117, 100, 111, 97, 119, 101, 114])
        == "afejudoawer"
    )


@max_score(5)
def test_one_merge():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert len(tokenizer.vocab) == 129
    assert tokenizer.vocab[128] == "12"
    assert len(tokenizer.merge_rules) == 1
    assert tokenizer.merge_rules[(49, 50)] == 128


@max_score(5)
def test_two_merges():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert len(tokenizer.vocab) == 130
    assert tokenizer.vocab[129] == "123"
    assert len(tokenizer.merge_rules) == 2
    assert tokenizer.merge_rules[(49, 50)] == 128
    assert tokenizer.merge_rules[(128, 51)] == 129


@max_score(5)
def test_100_merges():
    shakespear = "\n".join(
        ds.load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)["train"][
            "text"
        ]
    )

    tokenizer = ASCIIBPETokenizer.from_data(shakespear, 100)
    assert tokenizer.merge_rules == {
        (101, 32): 128,
        (116, 104): 129,
        (116, 32): 130,
        (115, 32): 131,
        (100, 32): 132,
        (44, 32): 133,
        (111, 117): 134,
        (101, 114): 135,
        (105, 110): 136,
        (121, 32): 137,
        (97, 110): 138,
        (111, 114): 139,
        (58, 10): 140,
        (111, 32): 141,
        (101, 110): 142,
        (97, 114): 143,
        (10, 10): 144,
        (32, 129): 145,
        (111, 110): 146,
        (108, 108): 147,
        (104, 97): 148,
        (44, 10): 149,
        (105, 131): 150,
        (101, 115): 151,
        (46, 144): 152,
        (121, 134): 153,
        (32, 115): 154,
        (116, 141): 155,
        (101, 97): 156,
        (138, 132): 157,
        (111, 119): 158,
        (111, 102): 159,
        (32, 109): 160,
        (32, 119): 161,
        (32, 104): 162,
        (136, 103): 163,
        (111, 109): 164,
        (32, 97): 165,
        (129, 128): 166,
        (99, 104): 167,
        (115, 116): 168,
        (32, 98): 169,
        (110, 111): 170,
        (102, 139): 171,
        (105, 114): 172,
        (118, 128): 173,
        (115, 101): 174,
        (105, 129): 175,
        (145, 128): 176,
        (101, 133): 177,
        (108, 105): 178,
        (84, 104): 179,
        (147, 32): 180,
        (114, 101): 181,
        (97, 130): 182,
        (105, 109): 183,
        (101, 143): 184,
        (105, 116): 185,
        (115, 130): 186,
        (65, 110): 187,
        (73, 32): 188,
        (111, 111): 189,
        (103, 104): 190,
        (97, 116): 191,
        (105, 115): 192,
        (134, 114): 193,
        (101, 101): 194,
        (135, 32): 195,
        (39, 131): 196,
        (187, 132): 197,
        (108, 101): 198,
        (170, 130): 199,
        (109, 137): 200,
        (117, 114): 201,
        (59, 10): 202,
        (46, 10): 203,
        (153, 114): 204,
        (114, 97): 205,
        (114, 105): 206,
        (148, 130): 207,
        (108, 132): 208,
        (159, 32): 209,
        (117, 130): 210,
        (108, 97): 211,
        (114, 111): 212,
        (101, 132): 213,
        (105, 130): 214,
        (101, 131): 215,
        (69, 78): 216,
        (135, 128): 217,
        (100, 133): 218,
        (107, 128): 219,
        (117, 110): 220,
        (83, 140): 221,
        (73, 78): 222,
        (32, 100): 223,
        (121, 133): 224,
        (97, 131): 225,
        (97, 108): 226,
        (119, 175): 227,
    }
