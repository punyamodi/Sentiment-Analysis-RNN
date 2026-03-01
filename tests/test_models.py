import numpy as np
import pytest

from sentiment.models.bilstm import build as build_bilstm
from sentiment.models.lstm import build as build_lstm
from sentiment.models.registry import build_model
from sentiment.models.simple_rnn import build as build_simple_rnn

BASE_CONFIG = {
    "model": {
        "vocab_size": 500,
        "embedding_dim": 8,
        "maxlen": 10,
        "units": [16, 8],
        "dropout": 0.1,
    }
}


def _random_input(config):
    return np.random.randint(0, config["model"]["vocab_size"], (4, config["model"]["maxlen"]))


def test_simple_rnn_output_shape():
    model = build_simple_rnn(BASE_CONFIG)
    out = model.predict(_random_input(BASE_CONFIG), verbose=0)
    assert out.shape == (4, 1)


def test_simple_rnn_output_range():
    model = build_simple_rnn(BASE_CONFIG)
    out = model.predict(_random_input(BASE_CONFIG), verbose=0)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_lstm_output_shape():
    config = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "recurrent_dropout": 0.0}}
    model = build_lstm(config)
    out = model.predict(_random_input(config), verbose=0)
    assert out.shape == (4, 1)


def test_bilstm_output_shape():
    model = build_bilstm(BASE_CONFIG)
    out = model.predict(_random_input(BASE_CONFIG), verbose=0)
    assert out.shape == (4, 1)


def test_registry_simple_rnn():
    config = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "type": "simple_rnn"}}
    model = build_model(config)
    assert model is not None


def test_registry_lstm():
    config = {
        **BASE_CONFIG,
        "model": {**BASE_CONFIG["model"], "type": "lstm", "recurrent_dropout": 0.0},
    }
    model = build_model(config)
    assert model is not None


def test_registry_bilstm():
    config = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "type": "bilstm"}}
    model = build_model(config)
    assert model is not None


def test_registry_unknown_raises():
    config = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "type": "unknown_arch"}}
    with pytest.raises(ValueError):
        build_model(config)
