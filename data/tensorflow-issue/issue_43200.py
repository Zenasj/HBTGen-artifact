# tf.random.uniform((1, 10), dtype=tf.float32) ← Input shape inferred from usage x = np.zeros([batch_size, 10])

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Wrap a Dense layer with WeightNormalization, as in the reproducer snippet.
        # This layer internally has a tf.bool variable causing TensorBoard histogram issues.
        self.wn_dense = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1), input_shape=(10,))
    
    def call(self, inputs):
        return self.wn_dense(inputs)

def my_model_function():
    # Return an instance of MyModel, built but untrained
    model = MyModel()
    # Build model by calling once with sample input tensor (needed for TF graph tracing etc)
    dummy_input = tf.zeros((1, 10), dtype=tf.float32)
    _ = model(dummy_input)
    return model

def GetInput():
    # Return input tensor matching the expected input shape: (batch_size=1, 10 features)
    # Using float32 to match Dense layer input requirements
    return tf.random.uniform((1, 10), dtype=tf.float32)

