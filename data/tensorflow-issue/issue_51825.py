# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Classic convolution: groups=1
        self.classic_conv = tf.keras.layers.Conv2D(
            filters=18, kernel_size=3, padding='same', groups=1)
        self.classic_gap = tf.keras.layers.GlobalAveragePooling2D()
        self.classic_dense = tf.keras.layers.Dense(1)
        
        # Grouped convolution: groups=3
        self.grouped_conv = tf.keras.layers.Conv2D(
            filters=18, kernel_size=3, padding='same', groups=3)
        self.grouped_gap = tf.keras.layers.GlobalAveragePooling2D()
        self.grouped_dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        # Forward pass through classic conv branch
        classic_x = self.classic_conv(inputs)
        classic_x = self.classic_gap(classic_x)
        classic_out = self.classic_dense(classic_x)
        
        # Forward pass through grouped conv branch
        grouped_x = self.grouped_conv(inputs)
        grouped_x = self.grouped_gap(grouped_x)
        grouped_out = self.grouped_dense(grouped_x)
        
        # Compare outputs from classic and grouped conv branches
        # This comparison is illustrative: we compute the absolute difference
        # and return it as a numeric tensor.
        #
        # In the context of the issue, grouped conv on CPU errors in backprop.
        # Here, we encapsulate both models and show their output difference.
        diff = tf.abs(classic_out - grouped_out)
        
        # For practical usage, one could also return both outputs separately or
        # return a boolean by some threshold. We choose numeric diff here.
        return diff

def my_model_function():
    # Returns an instance of MyModel with randomly initialized weights
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (1, 256, 256, 3)
    # matching the input shape expected by MyModel
    return tf.random.uniform(shape=(1, 256, 256, 3), dtype=tf.float32)

