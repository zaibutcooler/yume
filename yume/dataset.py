from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from .tokenizer import Tokenizer
import torch
from .utils import dummy_logger

import tiktoken

class Trainset(Dataset):
    def __init__(self, batch_size=48, dataset_url="zaibutcooler/japanwiki-vault"):
        self.batch_size = batch_size
        self.dataset_url = dataset_url
        self.tokenizer = None
        self.train_data = None
        self.val_data = None

    def _load_dataset(self):
        loaded_dataset = load_dataset(self.dataset_url)
        self.text = loaded_dataset["train"]["text"]
        dummy_logger("Successfully loaded the dataset")

    def _tokenize(self, use_tiktoken=True):
        if use_tiktoken:
            enc = tiktoken.encoding_for_model("gpt-4")
            self.tokenizer = enc
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.load_pretrained()
            self.text = torch.utils.data.Dataset(self.text).map(lambda x: self.tokenizer.encode(x))


    def _prep_bin(self):
        # Split the dataset into training and validation sets
        train_size = int(0.99 * len(self.text))
        val_size = len(self.text) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.text, [train_size, val_size])

        # Save the tokenized data to binary files
        self._save_to_bin(self.train_data, "train.bin")
        self._save_to_bin(self.val_data, "val.bin")

    def _save_to_bin(self, data, filename):
        arr_len = np.sum([len(x) for x in data], dtype=np.uint64)
        dtype = np.object_  # Change dtype to np.object_
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for x in data:
            arr[idx:idx + len(x)] = [x]  # Wrap x in a list to convert to object array
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
        dummy_logger("Preparing the Bin")

        self._prep_bin()