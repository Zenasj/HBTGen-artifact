# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred as two features per example as in original examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple sequential-like model with:
        # - Dense with 10 units + ReLU activation
        # - Output layer with 1 unit + sigmoid activation for binary classification
        self.hidden = tf.keras.layers.Dense(10, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.hidden(inputs)
        return self.output_layer(x)

    def predict_classes(self, inputs):
        # Convenience method to mirror Sequential.predict_classes() behavior:
        # For binary classification with sigmoid output, threshold at 0.5
        probs = self(inputs, training=False)
        return tf.cast(probs > 0.5, dtype=tf.int32)

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile to enable fitting if needed
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

def GetInput():
    # Return a random input tensor compatible with model input: 
    # batch size 3 (same as example), 2 features inputs (matches original data)
    return tf.random.uniform((3, 2), dtype=tf.float32)

