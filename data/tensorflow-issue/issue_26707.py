# tf.random.uniform((B, 32), dtype=tf.float32)  ← The input is a batch of feature vectors of size 32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # This model replicates the example Sequential model from the issue
        # - Dense(128, relu)
        # - Dropout(0.2)
        # - Dense(10, softmax)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
        self.dropout = tf.keras.layers.Dropout(0.2, name='dropout_1')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax', name='dense_2')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        # Dropout behaves differently during training and inference
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor that matches the input expected by the model
    # Assuming batch size of 8 as a reasonable default
    return tf.random.uniform((8, 32), dtype=tf.float32)

