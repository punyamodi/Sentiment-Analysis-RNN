from tensorflow.keras import Model


def build_model(config: dict) -> Model:
    model_type = config["model"]["type"]

    if model_type == "simple_rnn":
        from sentiment.models.simple_rnn import build
    elif model_type == "lstm":
        from sentiment.models.lstm import build
    elif model_type == "bilstm":
        from sentiment.models.bilstm import build
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return build(config)
