from .yume import Yume, Trainset, Config

config = Config()

dataset = Trainset()

dataset.build_dataset()

yume = Yume(config)



# assert len(dataset.data) > 0

# yume.pretrain(dataset)

# yume.sample()

# optional
# yume.huggingface_login("your hf tokens")
# yume.save_pretrained("yume")
