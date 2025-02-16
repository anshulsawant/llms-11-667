"""Custom Dataset Builder for Common Crawl."""
import datasets
import homework 
from typing import List, Iterator, Tuple, Dict, Any

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "A single shard of Common Crawl."
_DATA_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz" 
 
class MiniCleanedCommonCrawl(datasets.GeneratorBasedBuilder):
    def _info(self) -> datasets.DatasetInfo:
        """
        Should return a DatasetInfo object describing <string> type values for a url and it's corresponding text.
        """
        features = datasets.Features(
            {
                "url": datasets.Value("string"),
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Should return a SplitGenerator object which downloads your data and creates train and validation splits.
        """        
        downloaded_files = dl_manager.download_and_extract(_DATA_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": downloaded_files},
            )
        ]

        return datasets.SplitGenerator(),
    
    def _generate_examples(self, filepaths: List[str]) -> Iterator[Tuple[Any, Dict[str, str]]]:
        """
        Streams raw data from the downloaded file and yields tuples consisting of a unique ID and the url/cleaned text.
        Should call the functions you defined in homework.py. 
        """
        print(f'Generating examples from {type(filepaths)}')
        filepath = filepaths
        for url, html_text in datasets.utils.tqdm(homework.utils.read_warc_file(filepath)):
            text = homework.html_to_text(html_text)
            cleaned_text = homework.clean_text(text)
            cleaned_nopii_text = homework.replace_pii(cleaned_text)
            if homework.heuristic_quality_filter(cleaned_nopii_text):
                yield url, {"url": url, "text": cleaned_nopii_text}

if __name__ == "__main__":   
    # Note: Calling load_dataset caches the processed dataset locally.
    # The default cache directory is ~/.cache/huggingface/datasets.
    # To force the dataset to be recreated, you should pass in the
    # additional argument download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    dataset = datasets.load_dataset(
        "mini_ccc_solution.py",
        "MiniCleanedCommonCrawl",
        trust_remote_code=True,
        split=datasets.Split.TRAIN)
    
    # Iterate over the first 100 examples.
    mini_dataset = dataset.select(range(100))
    for ex in mini_dataset:
        print(ex["url"])