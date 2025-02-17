import argparse
import re
import utils
from typing import Set, Dict
import bs4
import string
from tqdm import tqdm

BAD_WORD_LIST = 'bad_word_list.txt'

def compare(warc_file, wet_file, url, output_warc=False):
    warc = utils.read_warc_file_url(warc_file, url).decode()
    wet = utils.read_wet_file_url(wet_file, url).decode()
    text = html_to_text(warc)
    cleaned_text = clean_text(text)
    cleaned_nopii_text = replace_pii(cleaned_text)
    passes_check = heuristic_quality_filter(cleaned_nopii_text)
    print(url)
    print("Passes heuristic quality filter:", passes_check)
    print(cleaned_nopii_text)
    print("\n\n\n")
    print("Wet:\n", wet)
    if output_warc:
        print("\n\n\n")
        print("Warc:\n", warc)


def html_to_text(html: bytes) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    return bs4.BeautifulSoup(html, "html.parser").get_text()


def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    pattern1 = r"\d{3}-\d{2}-\d{4}"
    replacement1 = "XXX-XX-XXXX"
    pattern2 = r"\d{10}"
    replacement2 = "XXXXXXXXXX"
    return re.sub(pattern2, replacement2, re.sub(pattern1, replacement1, text))


def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    long_word = re.compile(r"[a-zA-Z0-9"+string.punctuation+"]{101,}")
    punct = re.compile("[" + string.punctuation + "]")

    def is_clean(p: str) -> bool:
        return len(re.findall(long_word, p)) == 0 and len(re.findall(punct, p)) > 0

    return "\n".join([p for p in text.split("\n") if is_clean(p)])


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    bad_words = retrieve_bad_words()  # Read the bad words list

    sep = string.punctuation + string.whitespace
    words = {
        word.lower() for word in re.split(r"[" + sep + "]", text) if word.split(sep)
    }
    # Check for bad words
    if len(bad_words.intersection(words)) > 0:
        return False

    characters = {c for c in text}
    # Check for punctuation
    if not any(char in string.punctuation for char in characters):
        return False

    # Check for non-whitespace characters
    if not any(not char.isspace() for char in characters):
        return False

    # Check for allowed character percentage
    allowed_chars = (
        "["
        + string.ascii_letters
        + string.digits
        + string.punctuation
        + string.whitespace
        + "]"
    )
    allowed_count = len(re.findall(allowed_chars, text))
    if allowed_count / len(text) < 0.8:
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", type=str, default="", help="Specify the path for your warc file."
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=None,
        help="Specify the number of records you want to parse"
        " (only used for debugging with smaller sets)",
    )
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--count_only", action="store_true")
    args = parser.parse_args()

    count = 0
    passed_count = 0
    if args.fname:
        for url, html_text in tqdm(utils.read_warc_file(args.fname, args.num_records)):
            count += 1
            text = html_to_text(html_text)
            cleaned_text = clean_text(text)
            cleaned_nopii_text = replace_pii(cleaned_text)
            passes_check = heuristic_quality_filter(cleaned_nopii_text)
            if passes_check:
                passed_count += 1
            if args.filter and not passes_check:
                continue
            if args.count_only:
                continue
            print(url)
            print("Passes heuristic quality filter:", passes_check)
            print(cleaned_nopii_text)
            print("\n\n\n")
        print(f"Total count: {count}, passed count: {passed_count}")
    else:
        print("Usage: python homework.py --fname data.warc")


def retrieve_bad_words():
    with open(BAD_WORD_LIST, 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)
