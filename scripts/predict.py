import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorflow.keras.models import load_model

from sentiment.data.loader import get_word_index
from sentiment.data.preprocessor import encode_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def predict(model, text: str, word_index: dict, maxlen: int = 200) -> dict:
    encoded = encode_text(text, word_index, maxlen=maxlen)
    prob = float(model.predict(encoded.reshape(1, -1), verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return {"label": label, "probability": prob, "confidence": confidence}


def parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Text to classify (omit for interactive mode)")
    parser.add_argument("--maxlen", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    word_index = get_word_index()

    if args.text:
        result = predict(model, args.text, word_index, maxlen=args.maxlen)
        print(f"\nText       : {args.text}")
        print(f"Sentiment  : {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}")
    else:
        print("Interactive mode  (type 'quit' to exit)\n")
        while True:
            text = input("Review: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict(model, text, word_index, maxlen=args.maxlen)
            print(f"  -> {result['label']}  ({result['confidence']:.2%} confidence)\n")


if __name__ == "__main__":
    main()
