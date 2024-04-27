from torch.utils.data import Dataset
from datasets import load_dataset


# TODO setup dataset
class Trainset(Dataset):
    def __init__(self, batch_size=48):
        self.loaded_data = load_dataset("zaibutcooler/animanga-vault")
        self.texts = self.loaded_data["train"]["raw"]
        self.data = self.loaded_data["train"]["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return []
