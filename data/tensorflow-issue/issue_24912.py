# tf.random.uniform((B, 30), dtype=tf.float32)  # Input shape inferred from original code: (batch_size, 30)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        # A simple dense layer with softmax activation to mimic original functionality
        self.dense = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        """
        Forward pass returning softmax class probabilities.
        Inputs: tensor of shape (batch, 30)
        Returns: tensor of shape (batch, num_classes)
        """
        return self.dense(inputs)


def my_model_function():
    # Instantiate the MyModel with default 10 classes
    model = MyModel(num_classes=10)

    # Compile the model similarly to original code:
    # - optimizer: GradientDescentOptimizer with learning rate 0.1
    # - loss: sparse categorical crossentropy
    # Note: tf.train.GradientDescentOptimizer is legacy in TF 2.x,
    # so use tf.keras.optimizers.SGD to be compatible with TF 2.20.0
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss)

    return model


def GetInput():
    # Return a batch of input of shape (batch_size, 30)
    # Use batch size of 4 as in batching in original dataset pipeline
    batch_size = 4
    input_tensor = tf.random.uniform(shape=(batch_size, 30), dtype=tf.float32)
    return input_tensor

