import torch
from torch import nn
import torch.nn.functional as F

class Yume(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(x):
        pass