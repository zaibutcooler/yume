import unittest
from yume import Tokenizer


class TestTokenizer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.tokenizer = Tokenizer()
        self.dummy_text = "馬鹿なこと言わないでよ"

    def test_encode(self):
        pass

    def test_decode(self):
        pass

    def test_train_encoder(self):
        pass


if __name__ == "__main__":
    unittest.main()
