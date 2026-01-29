# tf.random.uniform((B, 784), dtype=tf.float32) ← Inferred input shape for MNIST flattened images of size 28*28=784

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers matching the original MNIST simple dense model
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(784,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


def my_model_function():
    # Instantiate and compile the model as per original code
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random batch of inputs matching the input shape used for the model:
    # The input shape is (B, 784) where 784 = 28*28 flattened MNIST images.
    # We pick batch size B=32 as a reasonable default.
    batch_size = 32
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)


# Notes and Assumptions:
# - The model replicates the original MNIST Keras Sequential model from the issue,
#   rewritten as a subclassed tf.keras.Model named MyModel.
# - Input shape is flattened 28x28 grayscale images, i.e. shape (None, 784).
# - Activation functions and layer sizes are preserved.
# - Dropout layer is included and controlled by the `training` flag in call().
# - The given issue focuses on TFLite conversion and quantization problems,
#   but the model here is just the original float model as a tf.keras.Model subclass.
# - No quantization or TFLite conversion logic is included because that relates to
#   converter API usage outside the core model definition.
# - Input generation matches input shape and dtype expected by this model.
# - The model is compatible with TF 2.20 and can be JIT compiled with XLA like:
#   @tf.function(jit_compile=True)
#   def compiled_fn(x):
#       return model(x)

