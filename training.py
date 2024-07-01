from yume import Yume
from yume.dataset import Trainset
from yume.config import yume_medium, Config

# Initialize the dataset with the desired URL
dataset = Trainset(dataset_url="zaibutcooler/nihon-wiki")

# Build the dataset
dataset.build_dataset()

# Optional: Create a custom config if needed
# dummy_config = Config(...)

# Initialize the Yume model with a pre-defined medium configuration
yume = Yume(config=yume_medium)

# Pretrain the model with the dataset
yume.pretrain(dataset)

# Optional: Fine-tune the model with the dataset
# yume.fine_tune(dataset)

# Optional: Upload the model to Hugging Face
# yume.huggingface_login("your_hf_tokens")
# yume.save_pretrained("zaibutcooler/yume")