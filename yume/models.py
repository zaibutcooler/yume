import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
    def forward(self):
        pass

class MLP(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
    def forward(self):
        pass

class Block(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
    def forward(self):
        pass
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(x):
        pass