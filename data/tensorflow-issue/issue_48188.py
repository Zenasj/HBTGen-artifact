# tf.ragged.constant with inner ragged dimensions of varying lengths, shape roughly (batch=2, None, 2)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Placeholder simple layer to demonstrate model behavior
        self.dense = tf.keras.layers.Dense(1)

    def calc_loss(self, batch_in):
        # dummy operation: mean of values in the ragged tensor
        # ragged tensors support reduce_mean directly
        return tf.reduce_mean(batch_in - tf.constant(0, dtype=tf.float32))

    @tf.function
    def train_step(self, batch_in):
        with tf.GradientTape(persistent=True) as tape:
            prediction_loss = self.calc_loss(batch_in)
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(prediction_gradients, self.trainable_variables))
        self.add_loss(lambda: prediction_loss)
        return {"loss": prediction_loss}

    def call(self, inputs, training=False):
        # For simplicity, call the train_step during training
        if training:
            return self.train_step(inputs)
        else:
            # During inference, run a dummy layer on the flat values of ragged tensor
            # to demonstrate usage (in reality you'll have your own logic)
            # Convert ragged to flat tensor for Dense layer.
            flat_values = inputs.flat_values
            out = self.dense(flat_values)
            return out

def my_model_function():
    model = MyModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def GetInput():
    # Create a ragged tensor with irregular outer dimension shapes, matching example in the issue.
    # We batch 2 samples together, each with a sequence length randomly 1 or 2 for two ragged components.
    seq_len_a = np.random.randint(1, 3)
    seq_len_b = np.random.randint(1, 3)

    a = np.zeros((seq_len_a, 2), dtype=np.float32)
    b = np.ones((seq_len_b, 2), dtype=np.float32)

    # Create ragged tensor with irregular row sizes
    ragged_tensor = tf.ragged.constant([a, b], dtype=tf.float32)
    # Return ragged tensor as input to model
    return ragged_tensor

