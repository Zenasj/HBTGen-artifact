# tf.random.uniform((B, ), dtype=tf.uint8) and tf.random.uniform((B, ), dtype=tf.float32)
# Inferred input shapes: input_a shape=(20,), dtype=np.uint8; input_b shape=(4,), dtype=tf.float32

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Configuration from the issue
        self.input_a_size = 20
        self.input_b_size = 4
        self.num_classes = 2
        self.len_embedding = 256
        
        # Layers for branch A (embedding + conv-based)
        self.embedding = tf.keras.layers.Embedding(self.len_embedding, 100)
        self.conv1d = tf.keras.layers.Conv1D(128, 4, activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling1D(4)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_a = tf.keras.layers.Dense(64, activation='relu')
        
        # Layers for branch B (fully connected)
        self.dense_b_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_b_2 = tf.keras.layers.Dense(32, activation='relu')
        
        # Concatenation and final classifier
        self.concat = tf.keras.layers.Concatenate()
        self.fc = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output')
        
    def call(self, inputs, training=False):
        input_a, input_b = inputs
        
        # Branch A
        x = self.embedding(input_a)
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        branch_a = self.dense_a(x)
        
        # Branch B
        y = self.dense_b_1(input_b)
        branch_b = self.dense_b_2(y)
        
        # Concatenate and classify
        concat_out = self.concat([branch_a, branch_b])
        fc_out = self.fc(concat_out)
        output = self.output_layer(fc_out)
        
        return output

def my_model_function():
    # Return instance of MyModel, suitable for compilation etc.
    return MyModel()

def GetInput():
    # Generate random inputs matching the functional API inputs
    # input_a: shape=(batch_size, 20), dtype=np.uint8 (integers for embedding input)
    # input_b: shape=(batch_size, 4), dtype=tf.float32 (continuous inputs)
    # We'll pick a moderate batch size (e.g. 32)
    batch_size = 32
    
    # input_a values must be integers in [0, len_embedding-1] because of Embedding layer
    input_a = tf.random.uniform(
        shape=(batch_size, 20),
        minval=0,
        maxval=256,
        dtype=tf.dtypes.uint8)
    
    # input_b values are floats
    input_b = tf.random.uniform(
        shape=(batch_size, 4),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32)
    
    return (input_a, input_b)

