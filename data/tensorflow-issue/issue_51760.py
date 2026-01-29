# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  # CIFAR-10 sized input images (batch size B is dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN model matching the CIFAR-10 example in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10)  # 10 classes CIFAR-10
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.output_layer(x)
        return x


def my_model_function():
    model = MyModel()
    
    # Compile model with SparseCategoricalCrossentropy with logits, Adam optimizer
    # This matches the original example in the issue
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random 4D tensor with shape (batch_size, 32, 32, 3)
    # Batch size chosen as 8 for example; dtype float32 per typical image tensor
    batch_size = 8  # Arbitrary but reasonable batch size
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

# ---
# ### Explanation / Assumptions:
# - The issue describes a small CNN model for CIFAR-10 with input shape `(32, 32, 3)`.
# - The model layers and structure are explicitly given in the initial chunks.
# - Loss is sparse categorical crossentropy from logits (no softmax applied in model output).
# - Optimizer used is Adam.
# - Input data is CIFAR-10 normalized between 0 and 1, so `tf.random.uniform` from 0 to 1 with shape `(batch_size, 32, 32, 3)` fits well.
# - Batch size in `GetInput()` is arbitrarily set to 8 for demonstration.
# - The main NCCL error relates to distributed training in SLURM `srun` environments and is not part of the model code itself.
# - Since the user wants a self-contained python model file, no distributed strategy code or usage environment setup is included.
# - Model is compatible with TensorFlow 2.20.0 and can be compiled with XLA (`@tf.function(jit_compile=True)`) if desired externally.
# - No test or run code included per instructions.
# If you want me to add a `@tf.function(jit_compile=True)` decorator around a compiled call or inference step, feel free to ask!