# Sentiment Analysis RNN

This repository contains code for a sentiment analysis model using Recurrent Neural Networks (RNNs) implemented in Python with Keras. The model aims to classify the sentiment of text data into positive or negative categories.

## Requirements

- Python 3
- Keras
- NumPy

## Model Architecture

The model architecture consists of the following layers:

1. **Embedding Layer**: Converts input text data into dense vectors of fixed size.
2. **Dropout Layer (0.4)**: Regularization layer to prevent overfitting by randomly setting input units to 0.
3. **SimpleRNN Layer (32)**: Recurrent layer with 32 units and returns sequences.
4. **Dropout Layer (0.4)**: Another regularization layer.
5. **SimpleRNN Layer (16)**: Recurrent layer with 16 units and does not return sequences.
6. **Dense Layer (1)**: Output layer with a sigmoid activation function for binary classification.

## Dataset

The model is trained on a dataset of text data with corresponding sentiment labels (positive or negative). Ensure that your dataset follows the required format for training the model.

## Training

To train the model, execute the `train.py` script after configuring the dataset path and other parameters as needed.

```bash
python train.py
```

## Evaluation

You can evaluate the trained model using the `evaluate.py` script by providing the path to the test dataset.

```bash
python evaluate.py --test_data_path <test_data_path>
```

## Model Summary

Below is the summary of the model architecture:

```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 50, 2)             10000     
_________________________________________________________________
dropout (Dropout)            (None, 50, 2)             0         
_________________________________________________________________
simple_rnn (SimpleRNN)       (None, 50, 32)            1120      
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
