from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import sys

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    return model

def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy')


def save_model_(model, log_dir, struct_only=False):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(log_dir / 'weights/weights')


def load_model_(log_dir):
    log_dir = Path(log_dir)
    with open(log_dir / 'model.json', 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(log_dir / 'weights/weights')
    return model


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")

    model = create_model()
    initial_learning_rate = 0.001
    compile_model(model, initial_learning_rate)

    model_path = 'my_model'
    save_model_(model, model_path)
    loaded_model = load_model_(model_path)

    new_learning_rate = 0.0001
    compile_model(loaded_model, new_learning_rate)
    print(f"Optimizer's learning rate of the loaded model: {loaded_model.optimizer.learning_rate.numpy():e}")
    # 1e-3
    compile_model(loaded_model, new_learning_rate)
    print(f"Optimizer's learning rate of the loaded model: {loaded_model.optimizer.learning_rate.numpy():e}")
    # 1e-4