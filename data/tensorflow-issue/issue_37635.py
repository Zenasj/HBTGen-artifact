# tf.random.uniform((B, ), dtype=tf.float32) ← Input is a 1D tensor of unspecified batch size, matching model build input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a BatchNormalization layer from normalization_v2 (TF2 default)
        # We explicitly use tf.keras.layers.BatchNormalization which resolves to normalization_v2.BatchNormalization
        self.bn_v2 = tf.keras.layers.BatchNormalization()
        # Create a BN layer from legacy normalization module
        # We simulate the legacy BN layer by importing directly from the 'normalization' module.
        # This simulates the difference described in the issue.
        # Since tf.keras does not expose normalization (v1) directly,
        # we create a similar layer by subclassing BatchNormalization to simulate the legacy behavior.
        # This is to demonstrate the difference discussed in the issue.
        class LegacyBatchNormalization(tf.keras.layers.BatchNormalization):
            # Override the class name for clarity (In TensorFlow internals it is different)
            @property
            def __class__(self):
                # Return a different class object to mimic type difference
                # Using type object with different qualified name
                class DummyClass(type(self)):
                    __qualname__ = "tensorflow.python.keras.layers.normalization.BatchNormalization"
                    __name__ = "BatchNormalization"
                return DummyClass
            
        self.bn_v1 = LegacyBatchNormalization()

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Run both batch norms on the same input
        out_v2 = self.bn_v2(inputs, training=training)
        out_v1 = self.bn_v1(inputs, training=training)

        # Return a boolean tensor comparing if types are the same as tf.keras.layers.BatchNormalization 
        # to illustrate the difference observed after save/load in the issue.
        # Since we simulated class difference in bn_v1, this will return False for bn_v1, True for bn_v2.
        is_v2_bn = tf.constant(isinstance(self.bn_v2, tf.keras.layers.BatchNormalization))
        is_v1_bn = tf.constant(not isinstance(self.bn_v1, tf.keras.layers.BatchNormalization))

        # Return out tensors and the boolean type checks (for demonstration)
        return out_v2, out_v1, is_v2_bn, is_v1_bn


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a 1D tensor compatible with model input_shape=(1,)
    # Use batch size 4 arbitrarily
    return tf.random.uniform((4, 1), dtype=tf.float32)

