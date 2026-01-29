# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape based on original example Input(shape=(1,), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple identity layer (mimicking the Lambda lambda x: x in the original issue)
        self.identity = tf.keras.layers.Lambda(lambda x: x)

        # Define a simple negation model as a submodel (mimicking the other_model)
        input_neg = tf.keras.Input(shape=(1,), dtype=tf.float32)
        output_neg = tf.keras.layers.Lambda(lambda x: -x)(input_neg)
        self.neg_model = tf.keras.Model(input_neg, output_neg)

        # Functional model input and output using the identity layer
        input_func = tf.keras.Input(shape=(1,), dtype=tf.float32)
        output_func = self.identity(input_func)
        self.func_model = tf.keras.Model(input_func, output_func)

    def call(self, inputs, training=False):
        # Run the functional model (identity)
        y_func = self.func_model(inputs, training=training)

        # Run the negation model on the same inputs
        y_neg = self.neg_model(inputs, training=training)

        # Compare outputs to demonstrate the problem described: 
        # we produce both outputs and return a boolean tensor indicating equality approx
        # This fuses the two submodels and their comparison into one model, as per instructions.

        # To compare, convert to float32 in case of dtype differences
        y_func_f = tf.cast(y_func, tf.float32)
        y_neg_f = tf.cast(y_neg, tf.float32)

        # Compute boolean tensor where outputs are close within tolerance
        is_close = tf.math.abs(y_func_f - y_neg_f) < 1e-5

        # For demonstration, return:
        # - the functional model output
        # - the negation model output
        # - the boolean tensor indicating if they are approximately equal elementwise
        return y_func, y_neg, is_close

def my_model_function():
    # Return an instance of MyModel, no special initialization necessary
    return MyModel()

def GetInput():
    # Return a random tensor input that matches input expected by MyModel
    # The input shape inferred from the issue is (batch_size, 1), float32 dtype
    batch_size = 4  # arbitrary batch size
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

