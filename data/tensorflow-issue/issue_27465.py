# tf.random.uniform((batch_size, feature_dim), dtype=tf.float32) ← input is expected to be a 2D tensor matching training_features length

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        # Simple sequential dense layers matching the reported example
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Assumptions:
    # - Inputs are vectors of length 10 (training_features length)
    # - Outputs are vectors of length 2 (prediction_features length)
    #
    # These are inferred from the example dense input shapes and output shapes.
    input_dim = 10
    output_dim = 2
    model = MyModel(input_dim, output_dim)
    # Build the model by calling it once (necessary for some TF internals)
    dummy_input = tf.random.uniform((1, input_dim), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Generate a random input tensor matching the model's input shape and dtype
    # Batch size set to 4 (arbitrary choice for testing)
    input_dim = 10
    batch_size = 4
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

