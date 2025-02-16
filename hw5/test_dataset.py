import mini_ccc
import datasets


def _load_dataset():
    mini_ccc._DATA_URL = "tiny_data.warc"
    dataset = mini_ccc.MiniCleanedCommonCrawl()
    dataset.download_and_prepare()
    return dataset


def test_dataloader_info():
    dataset = _load_dataset()
    assert len(dataset.info.description) > 0, "Description should be non-empty."


def test_dataloader_split():
    dataset = _load_dataset()
    result1 = "train" in dataset.info.splits
    result2 = datasets.Split.TRAIN in dataset.info.splits
    assert result1 or result2, "There should be one split, and it should be named 'train'."


def test_dataloader_iterate():
    dataset = _load_dataset()
    dataset = dataset.as_dataset(split='train')
    num_records = len(dataset)
    assert num_records > 0, "We expected your dataset to have at least 1 example in it, but instead it had 0."
