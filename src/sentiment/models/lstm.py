from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding


def build(config: dict) -> Sequential:
    cfg = config["model"]
    vocab_size: int = cfg["vocab_size"]
    embedding_dim: int = cfg["embedding_dim"]
    maxlen: int = cfg["maxlen"]
    units: list = cfg["units"]
    dropout: float = cfg["dropout"]
    recurrent_dropout: float = cfg.get("recurrent_dropout", 0.0)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Dropout(dropout))

    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        model.add(LSTM(u, return_sequences=return_seq, recurrent_dropout=recurrent_dropout))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))
    return model
