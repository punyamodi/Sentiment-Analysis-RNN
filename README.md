# Sentiment Analysis with RNN

Deep learning sentiment classifier for movie reviews using SimpleRNN, LSTM, and Bidirectional LSTM. Trained on the IMDB dataset (50 K reviews). Includes a training pipeline, evaluation suite, and an interactive Streamlit web app.

---

## Architecture

```
                     +---------------------------+
                     |      Raw Review Text      |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |  Tokenize & Pad (len=200) |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |      Embedding Layer      |
                     |  vocab=10000, dim=32/64   |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |          Dropout          |
                     +-------------+-------------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
 +----------v---------+  +---------v--------+  +---------v--------+
 |   SimpleRNN x2     |  |    LSTM x2       |  |   BiLSTM x2      |
 +----------+---------+  +---------+--------+  +---------+--------+
            |                      |                      |
            +----------------------+----------------------+
                                   |
                     +-------------v-------------+
                     |    Dense(1) + Sigmoid     |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     |    Negative / Positive    |
                     +---------------------------+
```

## Model Comparison

| Architecture | Parameters | Val Accuracy |
|:-------------|:----------:|:------------:|
| SimpleRNN    | ~329 K     | ~83 %        |
| LSTM         | ~788 K     | ~87 %        |
| BiLSTM       | ~1.1 M     | ~89 %        |

---

## Project Structure

```
sentiment-analysis-rnn/
├── src/
│   └── sentiment/
│       ├── data/
│       │   ├── loader.py          IMDB loading and word-index helpers
│       │   └── preprocessor.py    padding and text encoding
│       ├── models/
│       │   ├── registry.py        model factory
│       │   ├── simple_rnn.py
│       │   ├── lstm.py
│       │   └── bilstm.py
│       ├── training/
│       │   └── trainer.py         training loop with callbacks
│       ├── evaluation/
│       │   └── metrics.py         accuracy, F1, ROC-AUC, confusion matrix
│       └── utils/
│           ├── config.py          YAML config loader
│           └── visualization.py   training curves and confusion matrix plots
├── config/
│   ├── simple_rnn.yaml
│   ├── lstm.yaml
│   └── bilstm.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   └── app.py                     Streamlit inference app
├── tests/
│   ├── test_preprocessor.py
│   └── test_models.py
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## Dataset

The [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) contains 50 000 reviews split evenly between positive and negative sentiment.

| Split     | Samples |
|:----------|:-------:|
| Training  | 25 000  |
| Testing   | 25 000  |

The dataset is downloaded automatically by TensorFlow on first use via `tf.keras.datasets.imdb`.

---

## Installation

```bash
git clone https://github.com/punyamodi/Sentiment-Analysis-RNN.git
cd Sentiment-Analysis-RNN
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Train

```bash
python scripts/train.py --config config/lstm.yaml
python scripts/train.py --config config/bilstm.yaml
python scripts/train.py --config config/simple_rnn.yaml
```

Trained models are saved to `models/saved/`. Metrics and plots go to `results/`.

### Evaluate

```bash
python scripts/evaluate.py \
  --model-path models/saved/lstm_best.keras \
  --config config/lstm.yaml
```

### Predict

Single review:
```bash
python scripts/predict.py \
  --model-path models/saved/lstm_best.keras \
  --text "An outstanding film with brilliant performances."
```

Interactive mode:
```bash
python scripts/predict.py --model-path models/saved/lstm_best.keras
```

### Web App

```bash
streamlit run app/app.py
```

---

## Configuration

All hyperparameters live in YAML files under `config/`. Example:

```yaml
model:
  type: lstm
  vocab_size: 10000
  embedding_dim: 64
  maxlen: 200
  units: [128, 64]
  dropout: 0.4

training:
  optimizer: adam
  loss: binary_crossentropy
  epochs: 15
  batch_size: 128
  early_stopping_patience: 3
  reduce_lr_patience: 2
  reduce_lr_factor: 0.5
  min_lr: 1.0e-6
```

---

## Model Summaries

### SimpleRNN

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 200, 32)           320,000
dropout (Dropout)            (None, 200, 32)           0
simple_rnn (SimpleRNN)       (None, 200, 64)           6,208
dropout_1 (Dropout)          (None, 200, 64)           0
simple_rnn_1 (SimpleRNN)     (None, 32)                3,104
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 329,345
Trainable params: 329,345
Non-trainable params: 0
_________________________________________________________________
```

### LSTM

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 200, 64)           640,000
dropout (Dropout)            (None, 200, 64)           0
lstm (LSTM)                  (None, 200, 128)          98,816
dropout_1 (Dropout)          (None, 200, 128)          0
lstm_1 (LSTM)                (None, 64)                49,408
dropout_2 (Dropout)          (None, 64)                0
dense (Dense)                (None, 1)                 65
=================================================================
Total params: 788,289
Trainable params: 788,289
Non-trainable params: 0
_________________________________________________________________
```

### Original Baseline (legacy branch)

The original SimpleRNN model from v1 is preserved on the `legacy` branch for reference:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 50, 2)             10,000
_________________________________________________________________
dropout (Dropout)            (None, 50, 2)             0
_________________________________________________________________
simple_rnn (SimpleRNN)       (None, 50, 32)            1,120
_________________________________________________________________
dropout_1 (Dropout)          (None, 50, 32)            0
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 16)                784
_________________________________________________________________
dense (Dense)                (None, 1)                 17
=================================================================
Total params: 11,921
Trainable params: 11,921
Non-trainable params: 0
_________________________________________________________________
```

---

## Tests

```bash
pytest
```

---

## Requirements

| Package      | Version |
|:-------------|:--------|
| tensorflow   | >=2.12  |
| numpy        | >=1.23  |
| scikit-learn | >=1.2   |
| matplotlib   | >=3.6   |
| pyyaml       | >=6.0   |
| streamlit    | >=1.22  |
| pandas       | >=1.5   |

Python >= 3.8
