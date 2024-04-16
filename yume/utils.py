from .tokenizer import Tokenizer


# TODO setup utils
def encode(text):
    tokenizer = Tokenizer()
    result = tokenizer.encode(text)
    return result


def decode(tensor):
    tokenizer = Tokenizer()
    result = tokenizer.decode(tensor)
    return result


def load_data(data):
    pass
