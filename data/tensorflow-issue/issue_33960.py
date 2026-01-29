# tf.random.uniform((28000, 150, 27), dtype=tf.float32) ← inferred input shape from "28000x150x27 input from CSV"
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue, model has two LSTM layers and one Dense output layer with softmax activation
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(32, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel with no pretrained weights (as none were provided)
    return MyModel()

def GetInput():
    # Return an input tensor that matches the expected input shape for MyModel: batch size 1, sequence 150, features 27
    # The original dataset was described as 28000x150x27, but batch dimension is needed for inference, so batch=1 here.
    # Use float32 dtype matching common TF default
    batch_size = 1
    seq_length = 150
    feature_dim = 27
    return tf.random.uniform((batch_size, seq_length, feature_dim), dtype=tf.float32)

