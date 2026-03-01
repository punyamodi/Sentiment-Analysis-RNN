import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorflow.keras.models import load_model

from sentiment.data.loader import load_imdb
from sentiment.data.preprocessor import pad_data
from sentiment.evaluation.metrics import evaluate
from sentiment.utils.config import load_config
from sentiment.utils.visualization import plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved sentiment model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)

    logger.info("Loading test data")
    _, (X_test, y_test) = load_imdb(vocab_size=config["data"]["vocab_size"])
    X_test = pad_data(X_test, maxlen=config["data"]["maxlen"])

    metrics = evaluate(model, X_test, y_test)

    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("F1 Score : %.4f", metrics["f1_score"])
    logger.info("ROC-AUC  : %.4f", metrics["roc_auc"])
    logger.info("\n%s", metrics["classification_report"])

    model_name = Path(args.model_path).stem
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=str(results_dir / f"{model_name}_confusion_matrix.png"),
    )

    export = {k: v for k, v in metrics.items() if k not in ("y_prob", "y_pred")}
    with open(results_dir / f"{model_name}_eval.json", "w") as f:
        json.dump(export, f, indent=2)

    logger.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
