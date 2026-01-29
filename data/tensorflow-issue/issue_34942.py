# tf.random.uniform((B,)) ← The input is a dataset yield: two strings and a tensor of shape (100,120) float32; 
# for modeling a similar structure, we assume input tensor is of shape (100,120) float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Since the original issue is about tf.data shuffling and memory leaks,
        # there's no explicit model structure described.
        # We build a minimal model that takes input tensor and processes it.
        # For demonstration, apply a simple dense layer to last dim for demonstration.
        
        # Using TimeDistributed Dense to process 100x120 input as (100 time steps, 120 features)
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(64, activation='relu')
        )
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(10)  # example output dim = 10
    
    def call(self, inputs, training=False):
        # inputs is expected shape (B, 100, 120)
        x = self.dense(inputs)  # (B, 100, 64)
        x = self.global_pool(x)  # (B, 64)
        return self.classifier(x)  # (B, 10)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected input shape for MyModel
    # B (batch size) inferred from example: original ASRDataGenerator batches 192 samples,
    # but we choose a manageable batch size as 32 for demonstration.
    B = 32
    H = 100
    W = 120
    # Input dtype float32, tensor shape (B, H, W)
    return tf.random.uniform((B, H, W), dtype=tf.float32)

