"""Custom Dataset Builder for Common Crawl."""
import datasets
import homework 
from typing import List, Iterator, Tuple, Dict, Any

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = ""
_DATA_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz" 
 
class MiniCleanedCommonCrawl(datasets.GeneratorBasedBuilder):
    def _info(self) -> datasets.DatasetInfo:
        """
        Should return a DatasetInfo object describing <string> type values for a url and it's corresponding text.
        """
        return datasets.DatasetInfo(...)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Should return a List of SplitGenerator object which downloads your data and creates the train split.
        """
        return [datasets.SplitGenerator(...)]
    
    def _generate_examples(self, filepaths: List[str]) -> Iterator[Tuple[Any, Dict[str, str]]]:
        """
        Streams raw data from the downloaded file and yields tuples consisting of a unique ID and the url/cleaned text.
        The output should be the cleaned documents which pass the quality filter.
        Should call the functions you defined in homework.py and utils.py. 
        """
        pass

 
if __name__ == "__main__":   
    # Note: Calling load_dataset caches the processed dataset locally.
    # The default cache directory is ~/.cache/huggingface/datasets.
    # To force the dataset to be recreated, you should pass in the
    # additional argument download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    dataset = datasets.load_dataset(
        "mini_ccc.py",
        "MiniCleanedCommonCrawl",
        trust_remote_code=True,
        split=datasets.Split.TRAIN)
    
    # Iterate over the first 100 examples.
    for ex in dataset.take(100):
        print(ex["url"])