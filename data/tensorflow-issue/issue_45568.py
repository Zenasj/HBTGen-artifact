# tf.random.uniform((B, 28, 28, 1), dtype=tf.float16)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN model from the MNIST mixed precision example in the issue
        # Using mixed precision policy 'mixed_float16' globally
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', name='conv_1')
        self.maxpool_1 = tf.keras.layers.MaxPooling2D((2,2), name='maxpool_1')
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv_2')
        self.glob_maxpool = tf.keras.layers.GlobalMaxPooling2D()
        self.dense_logits = tf.keras.layers.Dense(10, name='logits')
        # Critical softmax output with explicit float32 dtype cast to avoid mixed precision instability
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.glob_maxpool(x)
        x = self.dense_logits(x)
        # Explicitly cast output to float32 to avoid numeric instability with mixed precision
        return self.softmax(x)

def my_model_function():
    # Set mixed precision policy globally before creating the model
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random batch tensor of shape (B, 28, 28, 1) with dtype float16
    # This matches the inputs used in the MNIST example with mixed precision
    batch_size = 32  # Made-up batch size for example; typical small batch
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float16)

