# tf.random.uniform((B, T, F), dtype=tf.float32) ← Assuming input is a 3D tensor with batch, time, features for masking layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Masking layer built to mask timesteps where all features == 1.0 (default mask_value=1.0)
        self.masking = tf.keras.layers.Masking(mask_value=1.0)
        # Dense layer to process masked input
        self.dense = tf.keras.layers.Dense(1)

        # We create a MeanSquaredError metric instance manually to show masking respects metric calculation
        self.metric = tf.keras.metrics.MeanSquaredError()

    def call(self, inputs, training=False):
        # Apply masking: this produces a masked tensor and sets the mask internally
        x = self.masking(inputs)

        # Compute dense layer output
        output = self.dense(x)
        return output

    def compute_mask(self, inputs, mask=None):
        # Forward mask from Masking layer, so downstream computations are aware
        return self.masking.compute_mask(inputs)

    def train_step(self, data):
        # Override train_step to apply mask to metrics manually (simulate pre-2.2 behavior)
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses,
                                      sample_weight=self.compute_mask(x))
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics but apply mask as sample weight to the metric calculation
        mask = self.compute_mask(x)
        mask_float = None
        if mask is not None:
            # Mask shape: (batch, time), metric expects sample_weight broadcastable to output shape (batch, time, 1)
            mask_float = tf.cast(mask, dtype=y_pred.dtype)
            # Expand mask dims if needed - y_pred may be (batch, time, 1)
            while len(mask_float.shape) < len(y_pred.shape):
                mask_float = tf.expand_dims(mask_float, axis=-1)

        # Manually update metric with mask applied as sample_weight, mimicking TF 2.1 behavior
        self.metric.update_state(y, y_pred, sample_weight=mask_float)
        self.compiled_metrics.update_state(y, y_pred)  # weighted_metrics handled internally in compile

        results = {m.name: m.result() for m in self.metrics}
        results['masked_metric'] = self.metric.result()  # expose manually tracked masked metric
        return {"loss": loss, **results}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected input shape
    # Assuming input shape (batch=2, timesteps=3, features=4) for demonstration and masking
    # This matches typical RNN input shapes where masking is commonly used
    import numpy as np
    B, T, F = 2, 3, 4
    # Create an input with some timesteps masked (features == 1.0)
    x = np.random.uniform(low=0.0, high=0.5, size=(B, T, F)).astype('float32')
    # Introduce some masked timesteps by setting all features to 1.0 for timestep 1 in batch 0
    x[0, 1, :] = 1.0
    return tf.convert_to_tensor(x)

