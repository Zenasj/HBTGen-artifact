# tf.random.uniform((B,)) where B is batch size, input features are scalar float32 per key: "age", "income"
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, features=("age", "income")):
        super().__init__()
        # Assuming features are scalar numeric columns as per the issue example
        self.feature_columns = [tf.feature_column.numeric_column(key) for key in features]
        # Create input placeholders for each feature key
        self.input_layers = {key: tf.keras.layers.Input(shape=(), name=key, dtype=tf.float32) for key in features}
        # DenseFeatures layer to integrate feature columns
        self.dense_features = tf.keras.layers.DenseFeatures(self.feature_columns)
        # Hidden Dense layer with 16 units (linear activation by default)
        self.dense_hidden = tf.keras.layers.Dense(16)
        # Output dense layer with 1 unit for binary classification (no activation)
        self.output_layer = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # inputs is expected to be a dict of tensors keyed by feature names
        x = self.dense_features(inputs)
        x = self.dense_hidden(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Instantiate MyModel with default features: "age", "income"
    model = MyModel()
    # Compile with Adam optimizer, binary crossentropy loss (from_logits=False),
    # no metrics (metrics were removed due to error in original issue with AUC on uniform labels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[],
    )
    return model

def GetInput():
    # Create batch size of 4 for testing
    batch_size = 4
    # Features: "age" and "income" both scalar float32
    # Provide a dictionary of tensors with shape (batch_size,) for each feature
    inputs = {
        "age": tf.random.uniform((batch_size,), dtype=tf.float32),
        "income": tf.random.uniform((batch_size,), dtype=tf.float32),
    }
    return inputs

