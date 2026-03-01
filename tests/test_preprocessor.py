import numpy as np
import pytest

from sentiment.data.preprocessor import encode_text, pad_data


def test_pad_data_output_shape():
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    result = pad_data(sequences, maxlen=5)
    assert result.shape == (3, 5)


def test_pad_data_truncates_long_sequences():
    sequences = [[1, 2, 3, 4, 5, 6, 7]]
    result = pad_data(sequences, maxlen=3)
    assert result.shape == (1, 3)


def test_pad_data_post_padding():
    sequences = [[1, 2]]
    result = pad_data(sequences, maxlen=4, padding="post")
    assert result[0, -1] == 0
    assert result[0, 0] == 1


def test_encode_text_output_shape():
    word_index = {"hello": 4, "world": 5, "<PAD>": 0, "<START>": 1, "<UNK>": 2}
    result = encode_text("hello world", word_index, maxlen=5)
    assert result.shape == (5,)


def test_encode_text_known_words():
    word_index = {"hello": 10, "world": 20, "<PAD>": 0, "<START>": 1, "<UNK>": 2}
    result = encode_text("hello world", word_index, maxlen=5)
    assert 10 in result
    assert 20 in result


def test_encode_text_unknown_words_map_to_unk():
    word_index = {"hello": 10, "<PAD>": 0, "<START>": 1, "<UNK>": 2}
    result = encode_text("hello unknownword", word_index, maxlen=5)
    assert 2 in result
