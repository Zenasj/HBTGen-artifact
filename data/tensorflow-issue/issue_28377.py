import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

py
def train_model(model, X_train, y_train, lr=0.01, epochs=1000, batch_size=32):

    training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    training_data = training_data.shuffle(buffer_size=2048).batch(batch_size)
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    def robust_mse(y_true, model_output):
        y_pred, var = tf.transpose(model_output)
        loss = 0.5 * tf.square(y_true - y_pred) * tf.exp(-var) + 0.5 * var
        return tf.reduce_mean(loss)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}/{epochs}")

        for X_batch, y_batch in training_data:

            with tf.GradientTape() as tape:
                output = model(X_batch)  # predictive mean and variance for each sample
                loss = robust_mse(y_batch, output)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

py
with open("features.csv") as file:
    features = np.genfromtxt(file, delimiter=",")
with open("labels.csv") as file:
    labels = np.genfromtxt(file, delimiter=",")

py
def build_model(input_dim, layers=None):
    """
    Build dropout FCNN with two outputs for aleatoric and epistemic uncertainty estimation.

    input_dim (int): number of features (columns in the matrix X)
    layers (tuple): overrides the default_layers below
    """
    default_layers = (
        ("Dense", {"units": 100, "activation": "tanh"}),
        ("Dropout", {"rate": 0.5}),
        ("Dense", {"units": 50, "activation": "relu"}),
        ("Dropout", {"rate": 0.3}),
        ("Dense", {"units": 25, "activation": "relu"}),
        ("Dropout", {"rate": 0.3}),
        ("Dense", {"units": 10, "activation": "relu"}),
        ("Dropout", {"rate": 0.3}),
    )
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,), name="input"))

    for layer, kwargs in layers or default_layers:
        model.add(getattr(tf.keras.layers, layer)(**kwargs))

    # predictive mean and data-dependent uncertainty output
    model.add(tf.keras.layers.Dense(units=2, activation="linear", name="output"))
    return model