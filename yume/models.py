import torch
from torch import nn
import torch.nn.functional as F
from .config import Config
from .utils import encode, decode


# TODO setup models
class SelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, x):
        pass


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, x):
        pass


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, x):
        pass


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        pass

    def _init_weights(self):
        pass

    def configure_optimizer(self):
        pass

    def generate(self):
        pass
