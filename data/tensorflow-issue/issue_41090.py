# tf.random.uniform((B, T, D), dtype=tf.float32)  # B=batch size, T=sequence length=5, D=embedding dim=5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Embedding layer to match the example in the issue:
        # input tokens of length 10 with vocab size 20, embedding dim 5
        self.embedding = tf.keras.layers.Embedding(input_dim=20, output_dim=5)
        self.my_layer = MyLayer1()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.embedding(inputs)  # (B, 10, 5)
        x = self.my_layer(x)        # custom segment_prod logic
        x = self.pooling(x)         # global average pooling over sequence dimension
        return x

class MyLayer1(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: (B, T=10, D=5)
        # segments to group features in dimension T=5 groups (example: 0,0,0,1,1)
        # The issue describes segment_prod over feature dims, so transpose dims accordingly
        # Note: The example used segments=[0,0,0,1,1] with length 5,
        # but input sequence length is 10, so let's adjust segments to length 10 accordingly.
        # Assuming splitting 10 timesteps into 2 groups of 5 each for demo.
        
        segments = tf.constant([0,0,0,0,0,1,1,1,1,1])
        # transpose inputs so that time dimension T is first for segment_prod over T
        # inputs shape: (B,T,D)
        x_t = tf.transpose(inputs, perm=[1, 0, 2])  # (T, B, D)
        # segment_prod needs T segments and performs product over segmented elements along axis 0
        
        # tf.math.segment_prod does not have a registered gradient in some TF versions.
        # An alternative is to use unsorted_segment_prod, which supports gradients.
        # This aligns with the issue's observation.
        prod = tf.math.unsorted_segment_prod(x_t, segments, num_segments=2)  # (2, B, D)
        
        # transpose back so output shape is (B, 2, D)
        output = tf.transpose(prod, perm=[1, 0, 2])
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape matches the input layer in MyModel:
    # Integer token ids between 0 and 19, shape (batch_size=4, sequence_length=10)
    B = 4
    T = 10
    return tf.random.uniform(shape=(B, T), minval=0, maxval=20, dtype=tf.int32)

