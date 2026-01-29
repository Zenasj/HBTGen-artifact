# tf.random.uniform((B, 10, 10, 1), dtype=tf.float32) ← Inferring input shape (10,10,1) as per example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We replicate the model described:
        # Input (10,10,1) -> clip_by_value -> Dense(8, activation='relu')
        # Note: Since clip_by_value is a tensorflow op, not a layer,
        # we implement it inside call().
        self.dense = tf.keras.layers.Dense(8, activation='relu')

    def call(self, inputs, training=False):
        # Apply clip_by_value before passing to dense layer
        clipped = tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.0)
        output = self.dense(clipped)
        return output


def my_model_function():
    # Instantiate the model and compile as per original example
    model = MyModel()
    # Create dummy input to build model weights
    dummy_input = tf.random.uniform((1, 10, 10, 1), dtype=tf.float32)
    model(dummy_input)

    # We compile the model with a dummy loss and Adam optimizer
    # Note: The original example used binary_crossentropy for output shape (None,8)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Add a custom metric which includes clip_by_value inside
    def custom_metric(y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1.0)
        return tf.reduce_mean(y_pred_clipped)

    # Add custom metric to the model via add_metric
    # To mimic the original usage, we pass model output directly as y_pred
    model.add_metric(custom_metric(model.output), name='val_custom', aggregation='mean')

    return model


def GetInput():
    # Return a random tensor of shape (1, 10, 10, 1) matching the input expected by MyModel
    return tf.random.uniform((1, 10, 10, 1), dtype=tf.float32)

