# tf.random.uniform((B, 10), dtype=tf.float32) ← input shape inferred from example input shape [10]

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Define a simple baseline model architecture matching the issue examples
        self.dense1 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=4, activation=None)
        
        # Define the same model but with L2 kernel regularizers set during init (like model_b)
        self.reg_dense1 = tf.keras.layers.Dense(units=16, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))
        self.reg_dense2 = tf.keras.layers.Dense(units=16, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))
        self.reg_dense3 = tf.keras.layers.Dense(units=4, activation=None,
                                                kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))
        
        # Define another model where regularizers are set post-hoc (like model_c)
        # We build layers first without regularizer, then assign kernel_regularizer attribute.
        # Note: This does NOT automatically add the losses in Keras (the issue reported)
        self.posthoc_dense1 = tf.keras.layers.Dense(units=16, activation='relu')
        self.posthoc_dense2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.posthoc_dense3 = tf.keras.layers.Dense(units=4, activation=None)
        
        # After building layers, manually set kernel_regularizer attributes (doesn't add to losses)
        for layer in [self.posthoc_dense1, self.posthoc_dense2, self.posthoc_dense3]:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(l=1e-5))
        
        # To emulate model_c's behavior, no losses added automatically here
        
    def call(self, inputs, training=False):
        # Forward pass through all three variants
        
        # model_a: simple model without regularization
        x_a = self.dense1(inputs)
        x_a = self.dense2(x_a)
        out_a = self.dense3(x_a)
        
        # model_b: model with regularization added during layer creation
        x_b = self.reg_dense1(inputs)
        x_b = self.reg_dense2(x_b)
        out_b = self.reg_dense3(x_b)
        
        # model_c: model with regularization set post-hoc (won't contribute losses)
        x_c = self.posthoc_dense1(inputs)
        x_c = self.posthoc_dense2(x_c)
        out_c = self.posthoc_dense3(x_c)
        
        # Now, emulate the comparison logic suggested by the issue and comments:
        # Compare model losses:
        # model_a.losses should be [] (no losses)
        losses_a = self._get_losses_dummy(out_a)  # model_a no losses expected
        
        # model_b.losses should contain regularization terms
        losses_b = self._get_losses_regularized()
        
        # model_c.losses would be empty because setting kernel_regularizer post-build
        losses_c = self._get_losses_posthoc()
        
        # Build outputs in a dictionary to analyze differences
        # Just return tensor combining key indicators:
        
        # For demonstration: output a vector concatenating
        # - sum of outputs from each model branch
        # - count of losses for each model version (as float)
        
        sum_out_a = tf.reduce_sum(out_a, axis=-1, keepdims=True)
        sum_out_b = tf.reduce_sum(out_b, axis=-1, keepdims=True)
        sum_out_c = tf.reduce_sum(out_c, axis=-1, keepdims=True)
        
        # Count losses approximated by lengths of lists (converted to float tensor)
        len_losses_a = tf.constant(float(len(losses_a)), shape=[1,1], dtype=tf.float32)
        len_losses_b = tf.constant(float(len(losses_b)), shape=[1,1], dtype=tf.float32)
        len_losses_c = tf.constant(float(len(losses_c)), shape=[1,1], dtype=tf.float32)
        
        # Concatenate all outputs / losses info on last dimension
        # shape: (batch_size, 6)
        output = tf.concat([sum_out_a, sum_out_b, sum_out_c,
                            len_losses_a * tf.ones_like(sum_out_a),
                            len_losses_b * tf.ones_like(sum_out_a),
                            len_losses_c * tf.ones_like(sum_out_a)], axis=-1)
        
        return output
    
    def _get_losses_dummy(self, _):
        # model_a: no regularizers, so no losses
        return []
    
    def _get_losses_regularized(self):
        # model_b: as layers have regularizers, collecting their losses
        # losses are collected automatically if layers created with kernel_regularizer param
        losses = []
        for layer in [self.reg_dense1, self.reg_dense2, self.reg_dense3]:
            if hasattr(layer, 'losses'):
                losses.extend(layer.losses)
        return losses
    
    def _get_losses_posthoc(self):
        # model_c: layers had kernel_regularizer attribute set post-build,
        # but Keras doesn't automatically add the regularization losses.
        # This is the core issue shown by the GitHub thread.
        losses = []
        for layer in [self.posthoc_dense1, self.posthoc_dense2, self.posthoc_dense3]:
            if hasattr(layer, 'losses'):
                losses.extend(layer.losses)
        return losses


def my_model_function():
    # Returns an instance of MyModel with initialized layers.
    return MyModel()


def GetInput():
    # Returns input tensor matching input shape expected (batch dimension included)
    # According to example input: shape=[10], so generate a batch of size 4 for versatility
    batch_size = 4
    input_shape = (batch_size, 10)
    return tf.random.uniform(input_shape, dtype=tf.float32)

