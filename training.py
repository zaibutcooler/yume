from .yume import Yume, Trainset, Config

config = Config()

dataset = Trainset()

dataset._load_dataset()

dataset._tokenize(tiktoken=True)

yume = Yume(config)

assert len(dataset.data) > 0

yume.pretrain(dataset.data)

yume.sample()

# optional
# yume.huggingface_login("your hf tokens")
# yume.save_pretrained("yume")
