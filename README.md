# Yume (夢)

Yume is a Japanese LLM (Large Language Model) with 1.5 billion parameters, inspired by Andrej Karpathy. It is trained on dialogues from anime and manga, aimed at generating anime dialogues. Future plans include creating a better version, [Yuumi](https://github.com/zaibutcooler/yuumi), which will be a lightweight LLM for daily tasks.

## Features

- Large language model for Japanese
- Trained on anime and manga dialogues
- Configurable with various model sizes
- Supports pretraining and fine-tuning
- Integrates with Hugging Face for model management

## Usage

### Sampling Text

You can use Yume to generate text samples. Here's an example:

```python
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
```

### Training the Model

You can also train Yume with your own dataset. Here’s how you can do it:

```python
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
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by Andrej Karpathy and utilizes dialogues from various anime and manga sources for training.

## Links

- [Source Code](https://github.com/zaibutcooler/yume/)
- [Opensourced Model](https://huggingface.co/zaibutcooler/yume/)
- [Hugging Face Space](https://huggingface.co/spaces/zaibutcooler/yume)
- [Japanese Dataset](https://huggingface.co/datasets/zaibutcooler/nihon-wiki)
- [Animanga Dataset](https://huggingface.co/datasets/zaibutcooler/animanga-dialogs)
