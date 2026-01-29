# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← Input shape for batch_size=1, steps=28, input_size=28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a bidirectional LSTM layer with 20 units returning sequences
        self.bidir_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(20, return_sequences=True))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')

    def call(self, inputs, training=False):
        x = self.bidir_lstm(inputs)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Instantiate the model and compile it with adam optimizer.
    model = MyModel()
    # The original example set up metrics and loss, prepare for categorical classification
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random float32 tensor with the shape expected by MyModel:
    # batch_size=1, steps=28, input_size=28
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

