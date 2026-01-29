# tf.random.uniform((8, 11), dtype=tf.float32) ← Input shape inferred from dataset generator: (max_seq_len=8, channels_count=11)
import tensorflow as tf

max_seq_len = 8
channels_count = 11

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Two Dense layers producing different sized outputs keyed by "A" and "B"
        self.dense_a = tf.keras.layers.Dense(3)  # output shape: (batch, seq_len, 3)
        self.dense_b = tf.keras.layers.Dense(4)  # output shape: (batch, seq_len, 4)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Expect input to be dict with key "F"
        seq = inputs["F"]
        out_a = self.dense_a(seq)  # e.g. shape (batch, max_seq_len, 3)
        out_b = self.dense_b(seq)  # e.g. shape (batch, max_seq_len, 4)
        return {"A": out_a, "B": out_b}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input matching expected input: dict with key "F" and shape (batch, 8, 11), float32
    batch_size = 8  # typical batch size matching dataset example
    return {"F": tf.random.uniform((batch_size, max_seq_len, channels_count), dtype=tf.float32)}

