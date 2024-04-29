import torch
from torch import nn
import torch.nn.functional as F
from .config import Config
from .models import GPT
from huggingface_hub import login
from .utils import dummy_logger, training_logger


class Yume:
    def __init__(self, config: Config):
        assert config is not None
        super().__init__()
        self.gpt = GPT
        self.model = GPT(config=config)
        self.config = config

    def train(self):
        pass

    def save_pretrained(self, name="yume"):
        self.model.save_pretrained(name)
        self.model.push_to_hub(name)
        dummy_logger("Successfully saved the pretrainied")

    def load_pretrained(self, url="zaibutcooler/yume"):
        self.model = self.gpt.from_pretrained(url)
        dummy_logger("Successfully loaded the pretrained")

    def huggingface_login(self, token):
        assert token is not None
        login(token=token)
        dummy_logger("Logged in successfully")

    def generate(self):
        pass
