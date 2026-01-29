# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from model input: (None, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        # A simple Dense layer as used in the example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

    def unpack_x_y_sample_weight(self, data):
        """Unpacks user-provided data tuple to x, y, and sample_weight."""
        if not isinstance(data, tuple):
            return (data, None, None)
        elif len(data) == 1:
            return (data[0], None, None)
        elif len(data) == 2:
            return (data[0], data[1], None)
        elif len(data) == 3:
            return (data[0], data[1], data[2])

    def train_step(self, data):
        x, y, sample_weight = self.unpack_x_y_sample_weight(data)

        # Custom print to verify execution of custom train_step
        tf.print("HIHI! I'm in function train_step!")

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        # Minimize loss using optimizer and tracked trainable variables
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    # Create an input layer with shape (10,)
    inputs = tf.keras.layers.Input(shape=(10,))
    # Build the MyModel network around inputs and outputs
    model = MyModel(inputs=inputs, outputs=MyModel().dense(inputs))
    return model

def GetInput():
    # Return a batch of random inputs matching input shape (batch=8 assumed)
    # Using float32 dtype consistent with typical TensorFlow default
    batch_size = 8
    return tf.random.uniform((batch_size, 10), dtype=tf.float32)

