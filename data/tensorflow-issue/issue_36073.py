# tf.random.uniform((B, 1), dtype=tf.float32) ← Inferring input shape as (batch_size, 1) from provided sample input (1,1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer with kernel initializer "ones" and trainable set False as in original code
        self.dense = tf.keras.layers.Dense(100, kernel_initializer="ones", trainable=False)
        # Dropout layer with rate 0.5
        self.dropout = tf.keras.layers.Dropout(0.5)
    
    def call(self, inputs, training=False):
        # Pass inputs through dense layer
        x = self.dense(inputs)
        
        # Apply dropout only during training
        if training:
            x = self.dropout(x, training=training)
        
        # Reduce sum of all elements in x, normalize by 100 to get average,
        # then reshape to [1,1] as per original example to avoid runtime issues
        x = tf.reshape(tf.reduce_sum(x) / 100., [1, 1])

        return x

def my_model_function():
    # Return instance of MyModel
    return MyModel()

def GetInput():
    # Generate random input tensor with shape (1,1) and float32 matching the example x = np.ones((1,1), dtype=np.float32)
    return tf.random.uniform((1,1), dtype=tf.float32)

