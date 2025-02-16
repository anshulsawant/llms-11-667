import json 
from warcio.archiveiterator import ArchiveIterator
from typing import Optional, Iterator, Tuple

BAD_WORD_LIST = '.bad_word_list.txt'


def read_warc_file(fname: str, num_to_read: Optional[int] = None) -> Iterator[Tuple[str, str]]:
    """Parses a warc file.
    Args:
        fname (str): File name of the WARC file.
        num_to_read (int, optional): Number of records to read. Useful for debugging. If None, reads all records.

    Returns:
        An iterator yielding (URL, HTML content) tuples.
    """

    count = 0
    with open(fname, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if (record.rec_type == 'response') and (record.http_headers.get_header('Content-Type') == 'text/html'):
                url = record.rec_headers.get_header('WARC-Target-URI')
                html = record.content_stream().read()
                count += 1
                yield (url,  html)
            if num_to_read and count == num_to_read:
                break


def read_wet_file(fname, num_to_read=None):
    """Parses a wet file.
    Args:
        fname (str): File name of the WET file.
        num_to_read (int, optional): Number of records to read. Useful for debugging. If None, reads all records.

    Returns:
        An iterator yielding (URL, text) tuples.
    """

    count = 0
    with open(fname, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if (record.rec_type == 'conversion') and (record.rec_headers.get_header('Content-Type') == 'text/plain'):
                url = record.rec_headers.get_header('WARC-Target-URI')
                text = record.content_stream().read()
                count += 1
                yield (url,  text)
            if num_to_read and count == num_to_read:
                break


def retrieve_bad_words():
    with open(BAD_WORD_LIST, 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)