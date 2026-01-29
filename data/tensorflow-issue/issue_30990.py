# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← The input shape is (batch, 28, 28), single channel grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 64 units, input shape (28, 28)
        # return_sequences=False to output only last output
        self.lstm = tf.keras.layers.LSTM(units=64, return_sequences=False)
        # Dense layer to output 10-class logits with softmax activation
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with sparse categorical crossentropy loss and SGD optimizer as in the original example
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=[]
    )
    return model

def GetInput():
    # Generate a random tensor with shape (batch_size, 28, 28) matching the expected input of LSTM
    # Using batch_size=64 to match original training batch size
    batch_size = 64
    # The input dtype is float32 normalized in [0,1], so we can sample uniform random floats
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

