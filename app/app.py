import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from sentiment.data.loader import get_word_index
from sentiment.data.preprocessor import encode_text

MODEL_OPTIONS = {
    "LSTM": "models/saved/lstm_best.keras",
    "BiLSTM": "models/saved/bilstm_best.keras",
    "SimpleRNN": "models/saved/simple_rnn_best.keras",
}
MAXLEN = 200

BASE_DIR = Path(__file__).parent.parent


@st.cache_resource
def load_resources(model_path: str):
    full_path = BASE_DIR / model_path
    if not full_path.exists():
        return None, None
    model = load_model(str(full_path))
    word_index = get_word_index()
    return model, word_index


def predict_sentiment(model, text: str, word_index: dict) -> dict:
    encoded = encode_text(text, word_index, maxlen=MAXLEN)
    prob = float(model.predict(encoded.reshape(1, -1), verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return {"label": label, "probability": prob, "confidence": confidence}


def main():
    st.set_page_config(
        page_title="Sentiment Analyzer",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("Sentiment Analyzer")
    st.write("Classify movie reviews as **Positive** or **Negative** using deep learning.")

    model_choice = st.selectbox("Model architecture", list(MODEL_OPTIONS.keys()))
    model, word_index = load_resources(MODEL_OPTIONS[model_choice])

    if model is None:
        st.error(
            f"Model not found at `{MODEL_OPTIONS[model_choice]}`. "
            "Train it first with:\n\n"
            f"```\npython scripts/train.py --config config/{model_choice.lower()}.yaml\n```"
        )
        return

    text = st.text_area("Enter a movie review:", height=150, placeholder="Type your review here...")

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = predict_sentiment(model, text, word_index)

            col1, col2 = st.columns(2)
            col1.metric("Sentiment", result["label"])
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            st.progress(result["probability"])
            st.caption(f"Raw probability: {result['probability']:.4f}")

    st.divider()
    st.subheader("Try an example")

    examples = [
        "This movie was absolutely incredible. The acting was superb and the story kept me hooked from start to finish.",
        "Terrible film. Complete waste of time and money. The plot made no sense whatsoever.",
        "A decent enough watch for a Friday night, nothing groundbreaking but entertaining.",
    ]

    for example in examples:
        label = example[:70] + "..."
        if st.button(label, key=example):
            if model is not None:
                result = predict_sentiment(model, example, word_index)
                st.info(f"**{result['label']}** — confidence {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
