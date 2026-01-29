# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape inferred from model input signature with batch dimension None and image size 28x28

import tensorflow as tf

IMG_SIZE = 28
NUM_CLASSES = 10

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a simple feedforward sequential model matching the original example
        # Flatten input 28x28 to vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        # Loss and optimizer are used in training function and stored here to reuse
        self._loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # from_logits=False because activation='softmax'
        self._optimizer = tf.keras.optimizers.SGD()

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),
    ])
    def train(self, x, y):
        # Custom training step for on-device training simulation
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self._loss_fn(y, logits)
        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Return dictionary storing loss and gradients keyed by variable names
        result = {"loss": loss}
        for var, grad in zip(self.trainable_variables, gradients):
            result[var.name] = grad
        return result

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32)])
    def predict(self, x):
        # Return output probabilities
        output = self(x, training=False)
        return {"output": output}

def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Generates a batch of random float inputs with shape (batch_size, 28, 28)
    # Here batch size is set to 4 for demonstration; it can be any positive integer.
    batch_size = 4
    return tf.random.uniform(
        shape=(batch_size, IMG_SIZE, IMG_SIZE),
        minval=0.0, maxval=1.0, dtype=tf.float32
    )

