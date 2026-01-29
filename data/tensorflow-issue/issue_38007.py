# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred MNIST input shape from usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST classifier as given in examples:
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with compiled optimizer and loss for classification
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

def GetInput():
    # Returns a batch of MNIST-like data: random float32 tensor in shape (B, 28, 28)
    # Using batch size 32 for example
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 28, 28), minval=0., maxval=1., dtype=tf.float32)

# ---
# ### Explanation / Context
# From the issue content:
# - The main minimal model tested extensively by the users was a simple MNIST classifier.  
# - Input shape was always `(28, 28)` grayscale images.
# - The common model was a `Sequential` with Flatten -> Dense 128 relu -> Dense 10 logits.
# - Single GPU usage works fine with this model.
# - Multi-GPU training via `tf.distribute.MirroredStrategy()` fails due to hardware/driver/BIOS issues (IOMMU setting).
# - The issue example code showcases construction and compilation of such a model and training on MNIST.
# - The model code here matches the minimal complete example used in the issue.
# - `GetInput()` generates appropriately shaped random inputs compatible with the model.
# - Returned model is compiled for training.
# - No multi-GPU strategy is implemented here, as the multi-GPU usage stalls because of the environment issues described. The model is single-instance, usable for both single-GPU or possibly multi-GPU (when the environment is fixed).
# - Input batch size chosen arbitrarily as 32 for demonstration.
# - This code matches TF 2.x API and should compile and run with TensorFlow 2.20.0 XLA without errors.
# If you want, you can wrap the model call inside `@tf.function(jit_compile=True)` for XLA compilation externally, for example:
# ```python
# model = my_model_function()
# @tf.function(jit_compile=True)
# def compiled(x):
#     return model(x)
# ```
# but that was not explicitly requested in output structure. The provided code is the minimal complete model and example input based on the issue content.
# ---
# Feel free to ask if you want me to help extend this to a multi-GPU strategy example or something else (noting the issue described environment limitations).