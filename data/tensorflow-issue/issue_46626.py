# tf.random.uniform((80, 28, 28, 1), dtype=tf.float32)  # inferred input shape and dtype from Fashion MNIST preprocessing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers from the Sequential model described
        # ThresholdedReLU with theta=0.3514439122821289
        self.thresh_relu = tf.keras.layers.ThresholdedReLU(theta=0.3514439122821289)
        # LeakyReLU with alpha=0.4855740853866919
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.4855740853866919)
        # AveragePooling2D with (1,2) pool size, padding same
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(1, 2), padding='same')
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer with num_classes=10 and softsign activation
        # Note: softsign is a nonlinear activation smooth alternative to tanh
        self.dense = tf.keras.layers.Dense(10, activation='softsign')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.thresh_relu(inputs)
        x = self.leaky_relu(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Since the original model is trained externally and saved,
    # here we return an untrained model instance.
    # Loading weights would require the saved file, omitted here per instructions.
    return model

def GetInput():
    # Return a random tensor matching the input expected by the model
    # The original dataset shape is (batch_size=80, height=28, width=28, channels=1),
    # and dtype float32 normalized [0,1]
    # We'll produce a random tensor with values in [0,1] range (like normalized images)
    batch_size = 80
    height, width, channels = 28, 28, 1
    return tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)

