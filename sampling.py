from .yume import Yume,Config

config = Config()

yume = Yume(config=config)

# Test the quality before loading the pretained
yume.sample()

yume.load_pretrained()

yume.sample()
