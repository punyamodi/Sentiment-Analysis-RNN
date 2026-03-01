import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def pad_data(
    sequences,
    maxlen: int = 200,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    return pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)


def encode_text(text: str, word_index: dict, maxlen: int = 200) -> np.ndarray:
    tokens = text.lower().strip().split()
    encoded = [word_index.get(token, word_index.get("<UNK>", 2)) for token in tokens]
    return pad_sequences([encoded], maxlen=maxlen, padding="post", truncating="post")[0]
