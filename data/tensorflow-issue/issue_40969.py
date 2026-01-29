# tf.random.uniform((None, 256, 256, 128, 1), dtype=tf.float32) ← Input shape inferred from the issue's example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Filters based on channel dimension in input shape (1)
        filters = 1
        # conv3d layer used in the residual block
        self.conv3d = tf.keras.layers.Conv3D(filters=filters, kernel_size=3, padding='same')
        
    def call(self, inputs):
        # Model replicates the "encoder" function with addition outside the residual block,
        # as this is the version that produces the "Connected to" column in model.summary().
        x = self.conv3d(inputs)
        outputs = x + inputs
        return outputs

def my_model_function():
    # Return an instance of MyModel with weights initialized by default
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape [None, 256, 256, 128, 1]
    # Here batch dimension is left as None, so generate a batch of 1 for testing
    input_shape = (1, 256, 256, 128, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

