# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred as (batch_size, 784) for a flattened 28x28 image vector

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim=10):
        super(MyModel, self).__init__(name="MyModel")
        self.output_dim = output_dim
        # Replicating the functional model architecture as submodules
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.dense_3 = tf.keras.layers.Dense(output_dim, name='predictions')

        # Submodel mimicking the "functional model" from issue for weight copying demonstration
        self.functional_layers = [
            tf.keras.layers.Dense(64, activation='relu', name='func_dense_1'),
            tf.keras.layers.Dense(64, activation='relu', name='func_dense_2'),
            tf.keras.layers.Dense(output_dim, name='func_predictions')
        ]

    def call(self, inputs):
        # Forward pass through the subclassed model path
        x_sub = self.dense_1(inputs)
        x_sub = self.dense_2(x_sub)
        x_sub = self.dense_3(x_sub)

        # Forward pass through the functional-like model path
        x_func = inputs
        for layer in self.functional_layers:
            x_func = layer(x_func)

        # Compare outputs numerically (L2 norm of difference)
        diff = tf.norm(x_sub - x_func, ord='euclidean', axis=-1, keepdims=True)

        # Output the difference tensor as numeric to reflect comparison
        # This shows how close the two submodels are in predictions
        return diff

def my_model_function():
    # Return an instance of MyModel with default output dimension 10
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size=1, 784) matching expected input
    # This simulates a flattened 28x28 image vector with float32 values
    return tf.random.uniform((1, 784), dtype=tf.float32)

