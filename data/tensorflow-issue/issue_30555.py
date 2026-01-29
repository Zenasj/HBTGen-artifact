# tf.random.uniform((4, 2), dtype=tf.uint8) ← Input shape inferred from XOR training_data input (4 samples, 2 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple XOR Keras model replicated from the issue:
        # - first Dense layer: 16 units, relu activation, no bias, input dim=2
        # - second Dense layer: 1 unit, sigmoid activation, no bias
        self.dense1 = tf.keras.layers.Dense(16, use_bias=False, activation='relu', input_shape=(2,))
        self.dense2 = tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights will be randomly initialized for now,
    # since original weights are from TF1 checkpoint, and conversion is outside this scope.
    return MyModel()

def GetInput():
    # Returns a batch of 4 XOR-like inputs matching the original input used.
    # Use uint8 as in the original numpy input, but convert to TF float32 for model input.
    xor_data = tf.constant([[0,0],[0,1],[1,0],[1,1]], dtype=tf.float32)
    return xor_data

