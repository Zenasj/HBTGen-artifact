# tf.random.uniform((None, 128, 256), dtype=tf.float32) ← inferred input shape from the issue input layer tf.keras.Input((128, 256))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Custom LSTM cell subclass that properly forwards kwargs to avoid errors
        class TestLSTMCell(tf.keras.layers.LSTMCell):
            def __init__(self, units, **kwargs):
                super(TestLSTMCell, self).__init__(units, **kwargs)

        self.cell = TestLSTMCell(512)
        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_state=True)
        self.bidirectional_layer = tf.keras.layers.Bidirectional(self.rnn_layer, merge_mode="concat")
    
    def call(self, inputs, training=False):
        # Forward pass through the bidirectional wrapper
        return self.bidirectional_layer(inputs, training=training)

def my_model_function():
    # Returns a new instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor matching the input shape expected by the model:
    # (batch_size, sequence_length=128, feature_dim=256)
    # batch size arbitrarily chosen as 4 for demonstration
    return tf.random.uniform((4, 128, 256), dtype=tf.float32)

