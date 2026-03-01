import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment.data.loader import load_imdb
from sentiment.data.preprocessor import pad_data
from sentiment.evaluation.metrics import evaluate
from sentiment.models.registry import build_model
from sentiment.training.trainer import train
from sentiment.utils.config import load_config
from sentiment.utils.visualization import plot_confusion_matrix, plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--name", type=str, default=None, help="Override model save name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model_name = args.name or config["model"]["type"]

    logger.info("Loading IMDB dataset (vocab=%d)", config["data"]["vocab_size"])
    (X_train, y_train), (X_test, y_test) = load_imdb(vocab_size=config["data"]["vocab_size"])

    maxlen = config["data"]["maxlen"]
    X_train = pad_data(X_train, maxlen=maxlen)
    X_test = pad_data(X_test, maxlen=maxlen)

    logger.info("Train shape: %s  Test shape: %s", X_train.shape, X_test.shape)

    model = build_model(config)
    model.summary()

    logger.info("Starting training")
    history = train(model, X_train, y_train, X_test, y_test, config, model_name)

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history,
        save_path=str(results_dir / f"{model_name}_training_curves.png"),
    )

    logger.info("Evaluating on test set")
    metrics = evaluate(model, X_test, y_test)

    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("F1 Score : %.4f", metrics["f1_score"])
    logger.info("ROC-AUC  : %.4f", metrics["roc_auc"])
    logger.info("\n%s", metrics["classification_report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=str(results_dir / f"{model_name}_confusion_matrix.png"),
    )

    export = {k: v for k, v in metrics.items() if k not in ("y_prob", "y_pred")}
    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(export, f, indent=2)

    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
