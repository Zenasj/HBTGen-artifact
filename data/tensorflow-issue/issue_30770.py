# tf.random.uniform((B, 218), dtype=tf.float32) ← inferred input shape is (?, 218) as per the discussion

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Explicitly define input dimension 218 as recommended to avoid shape issues
        self.dense1 = tf.keras.layers.Dense(218, activation="relu", input_shape=(218,))
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(436, activation="sigmoid")
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights are uninitialized until trained
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape expected by MyModel: (batch_size, 218)
    # Assuming batch size of 4 for example purposes
    batch_size = 4
    return tf.random.uniform((batch_size, 218), dtype=tf.float32)

