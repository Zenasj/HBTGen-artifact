# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original issue concerns the batch-dependent results in inference,
        # especially involving BatchNormalization layer in tf.keras.
        # To illustrate the problem and allow comparison,
        # we construct two submodels: one with BatchNorm, one without,
        # then compare their outputs on the same input batch sizes.
        
        self.batchnorm_branch = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=(128,128,1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        self.nobatchnorm_branch = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(128,128,1)),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
    def call(self, inputs, training=False):
        """
        Returns a dictionary with outputs from both branches, and a boolean tensor
        that indicates whether the first sample's outputs are equal between the two branches.
        Note that because of floating point numeric variance issues and batchnorm,
        outputs for the same sample can differ subtly.
        
        When called with a batch input tensor, the outputs and comparison refer
        to the first sample (index 0).
        """
        # Obtain outputs from the branch with BatchNorm in inference mode:
        # We force training=False so BatchNorm uses stored moving stats,
        # but note from the issue it still shows batch dependence on GPU.
        out_bn = self.batchnorm_branch(inputs, training=False)
        
        # Outputs from the branch without BatchNorm (direct flatten + dense)
        out_nobn = self.nobatchnorm_branch(inputs, training=False)
        
        # For comparison, check if outputs of first sample are close within tolerance
        equal = tf.reduce_all(
            tf.math.abs(out_bn[0] - out_nobn[0]) < 1e-5
        )
        
        return {
            'batchnorm_output': out_bn,
            'nobatchnorm_output': out_nobn,
            'first_sample_outputs_close': equal
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide a random uniform batch input of shape (batch_size, 128, 128, 1)
    # The batch size can be arbitrary; use 16 as a reasonable default
    batch_size = 16
    return tf.random.uniform((batch_size, 128, 128, 1), dtype=tf.float32)

