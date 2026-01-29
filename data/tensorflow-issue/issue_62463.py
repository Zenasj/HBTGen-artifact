# tf.random.uniform((10, 10, 10), dtype=tf.float32) ← inferred input shape and dtype from the original issue inputs

import tensorflow as tf

# Single model that fuses Model1 and Model2 to compare outputs under the hood,
# encapsulating the logic to highlight differences due to potential TF/XLA discrepancies.
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize parameters as variables matching original shapes:
        # p0 shape [1], p1 scalar []
        # Using initial random values within init, since original "params" array is from outside context
        self.p0 = tf.Variable(tf.random.uniform(shape=[1], dtype=tf.float32), trainable=False)
        self.p1 = tf.Variable(tf.random.uniform(shape=[], dtype=tf.float32), trainable=False)

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # inp shape: [10, 10, 10] float32

        # Model1 calculation: (p1 + p0) * inp - (p1 + p0) * inp
        sum_p = tf.add(self.p1, self.p0)          # shape broadcast scalar + vector (p1 scalar + p0 [1]) -> [1]
        model1_mul = tf.multiply(sum_p, inp)      # broadcasts p0+p1 over inp shape [10,10,10]
        model1_neg = tf.negative(model1_mul)
        model1_out = tf.add(model1_mul, model1_neg)

        # Model2 calculation: - (p0*inp + p1*inp) + p0*inp + p1*inp
        model2_p0_mul = tf.multiply(self.p0, inp)
        model2_p1_mul = tf.multiply(self.p1, inp)
        model2_sum = tf.add(model2_p1_mul, model2_p0_mul)
        model2_neg = tf.negative(model2_sum)
        model2_out = tf.add(tf.add(model2_neg, model2_p0_mul), model2_p1_mul)

        # Compare outputs from Model1 and Model2
        # Floating point parity check within a small tolerance (1e-6)

        # Calculate absolute difference tensor
        diff = tf.abs(model1_out - model2_out)
        # Are they close enough? Using tolerance slightly lower than original 0.01 to catch small numeric noise.
        tol = 1e-6
        close = tf.math.less_equal(diff, tol)

        # Return tuple with both outputs and a boolean tensor indicating elementwise closeness.
        # This captures the core of the original issue: subtle numeric difference despite semantic equality.
        return model1_out, model2_out, close

def my_model_function():
    # Return fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a valid random input tensor of shape [10, 10, 10], dtype float32 as expected by MyModel
    return tf.random.uniform(shape=[10, 10, 10], dtype=tf.float32)

