# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape based on example: batch size B, sequence length 28, feature size 28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The LSTM model as described: input (28,28), LSTM(20 units), Flatten, Dense(10 with softmax)
        self.lstm = tf.keras.layers.LSTM(20, time_major=False, return_sequences=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax', name='output')
    
    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel with uninitialized weights (user can train or load weights)
    model = MyModel()
    # Since original model compiled with sparse_categorical_crossentropy, compile similarly:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of inputs matching shape [1, 28, 28] and dtype float32 for forward pass
    # Matching the MNIST normalized float32 training input format
    import tensorflow as tf
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

