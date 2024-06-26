import torch
import torch.utils

class Config:
    def __init__(
        self,
        num_epoch: int,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        bias: bool,
        lr: float,
    ) -> None:
        self.num_epoch = num_epoch
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Small Yume model (around 100M parameters)
yume_small = Config(
    num_epoch=10,
    block_size=512,
    vocab_size=30522,
    n_layer=6,
    n_head=8,
    n_embd=256,
    dropout=0.1,
    bias=True,
    lr=0.001,
)

# Medium Yume model (around 500M parameters)
yume_medium = Config(
    num_epoch=10,
    block_size=1024,
    vocab_size=30522,
    n_layer=12,
    n_head=12,
    n_embd=512,
    dropout=0.1,
    bias=True,
    lr=0.001,
)

# Large Yume model (around 1B parameters)
yume_large = Config(
    num_epoch=10,
    block_size=2048,
    vocab_size=30522,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.1,
    bias=True,
    lr=0.001,
)