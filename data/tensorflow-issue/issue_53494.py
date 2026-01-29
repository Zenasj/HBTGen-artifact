# tf.random.uniform((1, 9), dtype=tf.float32) ← input shape inferred from coords2 tensor with shape (1, 9)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # gamma as a trainable variable
        self.gamma = tf.Variable(2.51412, dtype=tf.float32)

    def calc_sw3(self, gamma, gamma2, cutoff_jk):
        # placeholder calculation for energy, as per the example always returns 2.0
        E3 = 2.0
        return E3

    def calc_sw3_noerr(self, gamma0, cutoff_jk):
        # similar calculation without repeated argument
        E3 = 2.0
        return E3

    def energy(self, coords, gamma):
        # coords shape: (1, 9)
        xyz_i = coords[0, 0:3]  # First 3 elements
        xyz_j = coords[0, 3:6]  # Next 3 elements
        rij = xyz_j - xyz_i
        norm_rij = tf.sqrt(tf.reduce_sum(rij**2))
        # Original problematic call: repeated gamma arguments in tf.function:
        # E3 = self.calc_sw3(gamma, gamma, norm_rij)
        # To work around the tf.function issue, we implement both calls and output a dict comparing them.
        E3_with_repeat = self.calc_sw3(gamma, gamma, norm_rij)
        E3_no_repeat = self.calc_sw3_noerr(gamma, norm_rij)

        # Return both values and their difference for potential comparison
        return {
            'E3_with_repeat': E3_with_repeat,
            'E3_no_repeat': E3_no_repeat,
            'difference': E3_with_repeat - E3_no_repeat
        }

    def call(self, coords_all):
        return self.energy(coords_all, self.gamma)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate an input tensor matching the shape (1, 9) and dtype float32
    # This matches the coords tensor in the issue example
    return tf.random.uniform((1, 9), dtype=tf.float32)

