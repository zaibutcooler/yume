from torch.utils.data import Dataset
from datasets import load_dataset
from .tokenizer import Tokenizer
from .utils import dummy_logger

import tiktoken


# TODO setup dataset
class Trainset(Dataset):
    def __init__(self, batch_size=48):
        self.texts = None
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert len(self.data) > 10
        return []

    def _load_dataset(self, url="zaibutcooler/animanga-vault"):
        loaded_dataset = load_dataset(url)
        self.texts = loaded_dataset["animanga"]["texts"]
        dummy_logger("Successfully loaded the dataset")
        

    def _tokenize(self, tiktoken=True):
        if tiktoken:
            enc = tiktoken.get_encoding("cl100k_base")
            assert enc.decode(enc.encode("hello world")) == "hello world"

            enc = tiktoken.encoding_for_model("gpt-4")
            self.tokenizer = enc
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.load_pretrained()
        self.tokenizer.encode(self.texts)
        
    def _prep_bin(self):
        pass
    
    def get_batch(self):
        pass
    
    # from loading to installing in one function
    def build_dataset(self):
        pass