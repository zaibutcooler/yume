class Config:
    def __init__(
        self,
        num_epoch: int,
        block_sized=1024,
        vocab_sized=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        lr=0.001
    ) -> None:
        self.num_epoch = num_epoch
        self.block_sized = 1024
        self.vocab_sized = 50304
        self.n_layerd = 12
        self.n_headd = 12
        self.n_embdd = 768
        self.dropout = 0.0
        self.bias = True
        self.lr = lr
