from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from .tokenizer import Tokenizer
import torch
from .utils import dummy_logger

import tiktoken

class Trainset(Dataset):
    def __init__(self, batch_size=48, dataset_url="zaibutcooler/wiki-japanese"):
        self.batch_size = batch_size
        self.dataset_url = dataset_url
        self.tokenizer = None
        self.train_data = None
        self.val_data = None

    def _load_dataset(self):
        loaded_dataset = load_dataset(self.dataset_url)
        self.texts = loaded_dataset["animanga"]["texts"]
        dummy_logger("Successfully loaded the dataset")

    def _tokenize(self, tiktoken=True):
        if tiktoken:
            enc = tiktoken.get_encoding("cl100k_base")
            self.tokenizer = enc
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.load_pretrained()
        self.texts = self.texts.map(lambda x: self.tokenizer.encode(x))

    def _prep_bin(self):
        # Split the dataset into training and validation sets
        train_size = int(0.99 * len(self.texts))
        val_size = len(self.texts) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.texts, [train_size, val_size])

        # Save the tokenized data to binary files
        self._save_to_bin(self.train_data, "train.bin")
        self._save_to_bin(self.val_data, "val.bin")

    def _save_to_bin(self, data, filename):
        arr_len = np.sum([len(x) for x in data], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for x in data:
            arr[idx:idx + len(x)] = x
            idx += len(x)
        arr.flush()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index]

    def get_batch(self, batch_size):
        # Return a batch of examples from the training data
        return self.train_data[:batch_size]

    def build_dataset(self):
        self._load_dataset()
        self._tokenize()
        self._prep_bin()