# tf.random.uniform((10, 1), dtype=tf.int32) ← Based on inputs = np.arange(10) with shape (10,) and input_shape=[1] in InputLayer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimic the original Sequential model:
        # InputLayer(input_shape=[1]), Dense(1)
        # Note: InputLayer is implicit in Functional/Model subclassing, so no explicit InputLayer layer needed.
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # The forward pass corresponds to sequential passing through Dense
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model and compile to keep interface consistent with original
    model = MyModel()
    # Compile with the same settings as original example
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model

def GetInput():
    # Based on the original code, input shape was (10,), but model input_shape=[1], so input to model is shape (batch, 1)
    # Also, original inputs were integers (np.arange(10)) 
    # Since InputLayer was added with no dtype specified, default float32 will be expected by Dense
    # But issue references int dtype usage as reason for InputLayer necessity - we choose int32 inputs and let Dense cast.
    # To avoid mismatch, produce input tensor of shape (10,1) with dtype tf.int32
    
    # Create integer inputs shaped (10, 1)
    inputs = tf.random.uniform(shape=(10, 1), minval=0, maxval=10, dtype=tf.int32)
    return inputs

