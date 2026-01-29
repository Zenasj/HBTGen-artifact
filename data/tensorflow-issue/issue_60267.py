# tf.random.uniform((B, 30, 30), dtype=tf.int32)  ← Based on input shape (batch_size, 30, 30) from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from the issue's example sequential model
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        # Output units = number of classes, inferred as 51 from the issue example
        self.logits = tf.keras.layers.Dense(51)
        # A SparseCategoricalCrossentropy loss layer to compute loss internally (for comparison purpose)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, sample_weight=None, labels=None):
        """
        Forward pass returning logits.
        If labels and sample_weight are provided, return weighted loss as a scalar tensor,
        else return raw logits.
        
        This design reflects the scenario in the issue:
        - inputs: input tensor of shape (B, 30, 30)
        - sample_weight: sample weights for each input, shape (B,) or None
        - labels: sparse class labels, shape (B,) or None

        If sample_weight and labels are given, compute the weighted loss,
        else just output the logits.
        """
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.logits(x)

        if (labels is not None) and (sample_weight is not None):
            # Compute unweighted loss per sample
            unweighted_loss = self.loss_fn(labels, logits, sample_weight=None, reduction=tf.keras.losses.Reduction.NONE)
            # If sample_weight is provided, apply it per sample manually
            weighted_loss = unweighted_loss * tf.cast(sample_weight, unweighted_loss.dtype)
            # Return mean weighted loss scalar for validation or training step
            return tf.reduce_mean(weighted_loss)
        else:
            return logits

def my_model_function():
    # Return an instance of MyModel
    # This uses default initialization (weights initialized randomly)
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (B, 30, 30)
    # Here we simulate integer inputs as in the example, scaled uniformly between 300 and 900.
    # We pick an arbitrary batch size of 128 to match batch size from the issue example.
    batch_size = 128
    inp = tf.random.uniform(shape=(batch_size, 30, 30), minval=300, maxval=900, dtype=tf.int32)
    return inp

