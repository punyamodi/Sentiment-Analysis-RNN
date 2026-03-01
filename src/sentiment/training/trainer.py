import logging
from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

logger = logging.getLogger(__name__)


def get_callbacks(config: dict, model_name: str) -> list:
    cfg = config["training"]
    paths = config["paths"]

    model_dir = Path(paths["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(paths["log_dir"]) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_dir / f"{model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["reduce_lr_factor"],
            patience=cfg["reduce_lr_patience"],
            min_lr=cfg["min_lr"],
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
    ]


def train(model, X_train, y_train, X_val, y_val, config: dict, model_name: str):
    cfg = config["training"]

    model.compile(
        optimizer=cfg["optimizer"],
        loss=cfg["loss"],
        metrics=cfg.get("metrics", ["accuracy"]),
    )

    callbacks = get_callbacks(config, model_name)

    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    return history
