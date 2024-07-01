from yume import Yume
from yume.config import yume_small

# Optional: Create a custom config if needed
# dummy_config = Config(...)

# Initialize the Yume model with a pre-defined small configuration
yume = Yume(config=yume_small)

# Load a pretrained model from the specified path
yume.load_pretrained('zaibutcooler/yume')

# Generate a sample with the prompt '犬とは' (What is a dog?)
yume.sample('犬とは')