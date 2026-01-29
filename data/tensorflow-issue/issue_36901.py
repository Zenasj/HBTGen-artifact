# tf.random.uniform((1, 8, 8, 8, 1), dtype=tf.float32)  # Inferred input shape from example code in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # ConvLSTM2D layer as in minimal repro example
        self.convlstm = tf.keras.layers.ConvLSTM2D(
            filters=4,
            kernel_size=(2, 2),
            return_sequences=True,
            name='conv_lstm2d'
        )
        # MaxPooling3D with pool size (1, 2, 2) to downsample spatial dims only
        self.maxpool = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # Flatten layer to prepare for dense
        self.flatten = tf.keras.layers.Flatten()
        # Dense output layer to produce scalar output
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass through ConvLSTM2D
        x = self.convlstm(inputs, training=training)
        x = self.maxpool(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build the model by calling once (optional but helpful)
    # Assuming batch size 1 and input shape (8,8,8,1)
    dummy_input = tf.random.uniform((1,8,8,8,1), dtype=tf.float32)
    _ = model(dummy_input)
    return model

def GetInput():
    # Return a random tensor matching input expected by MyModel:
    # Shape: (batch_size=1, time=8, height=8, width=8, channels=1), float32
    return tf.random.uniform((1, 8, 8, 8, 1), dtype=tf.float32)

