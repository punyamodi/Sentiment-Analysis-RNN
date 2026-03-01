import numpy as np
from tensorflow.keras.datasets import imdb as keras_imdb


def load_imdb(vocab_size: int = 10000):
    (X_train, y_train), (X_test, y_test) = keras_imdb.load_data(num_words=vocab_size)
    return (X_train, y_train), (X_test, y_test)


def get_word_index(vocab_size: int = 10000) -> dict:
    raw_index = keras_imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in raw_index.items() if (v + 3) < vocab_size}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index
