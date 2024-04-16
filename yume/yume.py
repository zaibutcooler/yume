import torch
from torch import nn
import torch.nn.functional as F
from config import Config
from .models import GPT

config = Config()

class Yume:
    def __init__(self, config):
        super().__init__()
        self.model = GPT(config=config)

    def train(self):
        pass

    def load_pretrained(self):
        pass

    def generate(self):
        pass