# tf.random.uniform((B, 20), dtype=tf.float32) ← Input shape inferred from example code with input_shape=(20,)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple example model matching the example shape
        self.dense1 = layers.Dense(32, activation="relu", input_shape=(20,))
        self.dense2 = layers.Dense(20, activation="relu")
        self.dense3 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    """
    Instantiate and compile the model with the AUC and F1Score metrics as shown in the example.
    This setup reflects the issue context where EarlyStopping tries to monitor 'val_auc' but 
    metric suffixes can cause confusion in multiple model instantiations.
    """
    model = MyModel()
    # Compile the model: use binary crossentropy with AUC and F1Score metrics
    # F1Score is in TF 2.8+ under tf.keras.metrics but confirm available in TF2.20.0 as per requirements
    # If not available, fallback to dummy metric or custom implementation (here assume availability)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[metrics.AUC(name="auc"), metrics.F1Score(num_classes=1, average="micro", name="f1_score")]
    )
    return model

def GetInput():
    """
    Generate a random float tensor input with shape (batch_size, 20) matching the model's input_shape.
    Also generate dummy binary targets matching the output shape for testing compilation and fitting.
    """
    batch_size = 32  
    x = tf.random.uniform((batch_size, 20), dtype=tf.float32)
    y = tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.int32)  # binary labels
    y = tf.cast(y, tf.float32)
    return x, y

