import unittest
from yume import Yume,Config


class TestPretrained(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.config = Config()
        self.yume = Yume(config=self.config)
    
    def test_download(self):
        self.yume.load_pretrained()
        pass

    def test_generation(self):
        self.yume.sample()
        pass


if __name__ == "__main__":
    unittest.main()
