# tf.random.uniform((BATCH_SIZE, 100), dtype=tf.float32) ← Input shape inferred: (batch_size, 100) from example x=np.ones((10,100))

import tensorflow as tf

BATCH_SIZE = 2  # Used in example, assumed fixed batch size for demonstration

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A simple dense layer transforming input (shape [BATCH_SIZE, 100]) to output shape [BATCH_SIZE, 10]
        self.layer = tf.keras.layers.Dense(10)

        # Storing model output as a tf.Variable to enable extracting it outside train_step logs.
        # Shape is fixed to (BATCH_SIZE, 10) - assumes fixed batch size for demonstration.
        # This addresses the issue where train_step is tf.function and can't return arbitrary outputs.
        self.outputs = tf.Variable(tf.zeros((BATCH_SIZE, 10)), trainable=False)

    def call(self, inputs):
        return self.layer(inputs)

    @tf.function  # Ensures compatibility with tf.function and XLA compilation
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Update stored output variable
            self.outputs.assign(y_pred)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        # Return only metrics dict here, not the outputs, avoiding issues with callbacks treating them as scalars
        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    # Instantiate and return the model; user should compile it externally
    return MyModel()

def GetInput():
    # Generates random input matching the input expected by MyModel: (BATCH_SIZE, 100), dtype float32
    # Matches the example usage shape (10,100) with batch dimension, but fixed to BATCH_SIZE=2 here for consistency
    return tf.random.uniform((BATCH_SIZE, 100), dtype=tf.float32)

