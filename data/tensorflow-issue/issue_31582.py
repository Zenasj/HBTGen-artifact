# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← Inferred input shape for MNIST-like inputs as used in example

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the Sequential model architecture used in the example:
        # Conv2D, BatchNorm, Conv2D, BatchNorm ... ending with Dense softmax 10 classes
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    16, (3, 3),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu', input_shape=(28, 28, 1)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    32, (3, 3), strides=(2, 2),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    32, (3, 3),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    64, (3, 3), strides=(2, 2),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    64, (3, 3),
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu'
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Dense(
                    64, kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='relu'
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(
                    10, kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation='softmax'
                )
            ]
        )

        # Setup optimizer with MovingAverage wrapper (to reflect the example setup)
        base_optimizer = tfa.optimizers.LazyAdam(decay=1e-4)
        self.optimizer = tfa.optimizers.MovingAverage(base_optimizer)

        # Compile must be done outside __init__ usually, but 
        # we include here for completeness to match example intent
        self.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def assign_average_vars(self):
        # Assign EMA average weights to model variables
        # This duplicates the example usage of optimizer.assign_average_vars(model.variables)
        # Added extra check for compatibility to avoid None or unhashable variable issues.
        # The original issue was caused by variables being unhashable in dictionaries.
        # Here, we safely assign averages if they exist.
        var_list = [v for v in self.variables if self.optimizer._ema.average(v) is not None]
        assign_ops = [v.assign(self.optimizer._ema.average(v)) for v in var_list]
        return tf.group(assign_ops)

def my_model_function():
    model = MyModel()
    return model

def GetInput():
    # Produces a batch of 64 MNIST-like grayscale images normalized [0,1]
    return tf.random.uniform((64, 28, 28, 1), dtype=tf.float32)

