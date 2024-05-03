import unittest
from yume.dataset import Trainset


class TestDatasets(unittest.TestCase):
    def test_download(self):
        trainset = Trainset()
        trainset._load_dataset()
        assert trainset.texts is not None
        trainset._tokenize()
        assert len(trainset.data) > 1

    def test_encode(self):
        trainset = Trainset()
        dummy_text = "Hello Human World"
        trainset.texts = dummy_text
        trainset._tokenize()
        assert len(trainset.data) > 1
        encoded_text = trainset.tokenizer.encode(dummy_text)
        assert trainset.tokenizer.decode(encoded_text) == dummy_text


if __name__ == "__main__":
    unittest.main()
