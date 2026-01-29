# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the main structure discussed in the issue:
    Input (dynamic batch size, 28x28 grayscale images) ->
    Reshape to (28,28,1) ->
    Conv2D (filters=12, kernel_size=3x3) ->
    BatchNormalization ->
    ReLU ->
    MaxPooling2D ->
    Flatten ->
    Dense (10 logits)
    
    The issue focused on folding BatchNorm into Conv2D weights under quantization
    with per-channel vs per-tensor weight quantization.
    """
    def __init__(self):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))
        self.conv = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3))
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    """
    Returns an instance of MyModel.
    No pretrained weights are loaded because the issue did not provide them.
    """
    model = MyModel()
    # Model is untrained / default initialized
    return model

def GetInput():
    """
    Return a random input tensor with dynamic batch size and shape [B, 28, 28]
    to match Keras InputLayer shape used in the issue.
    
    Assumption:
    - Batch size = 4 (arbitrary example for dynamic batching)
    - Input dtype float32 normalized [0,1], as the MNIST inputs described.
    """
    batch_size = 4
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

