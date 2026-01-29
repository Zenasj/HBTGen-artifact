# tf.random.uniform((B, 1, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from the issue (batch size B is dynamic)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use mixed precision policy as specified in original issue (mixed_float16)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        # ResNet50 backbone inside a TimeDistributed wrapper to process the time dimension (sequence length = 1)
        # Note: The original issue used include_top=False, weights=None, pooling='avg'
        self.td_resnet = tf.keras.layers.TimeDistributed(
            tf.keras.applications.ResNet50(
                include_top=False, weights=None, pooling='avg'))
        
        # Remove the time dimension dimension via squeeze in call()
        self.squeeze_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))
        
        # Final Dense layer with 10 outputs (as in the example: logits for 10 classes)
        # Use dtype float32 for output logits to avoid dtype issues due to mixed precision
        self.dense = tf.keras.layers.Dense(10, dtype='float32')
        
        # Linear activation is effectively identity, explicitly set dtype to float32
        self.linear_activation = tf.keras.layers.Activation('linear', dtype='float32')
    
    def call(self, inputs, training=False):
        """
        inputs: (B, 1, 224, 224, 3) float32 tensor expected
        returns: (B, 10) logits tensor in float32
        """
        x = self.td_resnet(inputs)       # TimeDistributed ResNet50 output shape (B, 1, 2048)
        x = self.squeeze_layer(x)        # Remove the time dim --> (B, 2048)
        x = self.dense(x)                # Dense layer output (B, 10)
        x = self.linear_activation(x)   # Ensure output dtype float32
        return x

def my_model_function():
    # Return an instance of MyModel as required
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input shape (B, 1, 224, 224, 3).
    # Use batch size 5 as in the example.
    # Use float32 dtype inputs since this is standard for image inputs
    batch_size = 5
    input_shape = (batch_size, 1, 224, 224, 3)
    # Random values in [0,1), matching preprocessing in the example (divided by 255 and cast to float32)
    x = tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)
    return x

