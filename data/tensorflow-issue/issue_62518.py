# tf.random.uniform((B, 18), dtype=tf.float32), tf.random.uniform((34, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We encapsulate the two models from the issue as submodules, but here just as methods
        # No additional layers needed beyond TF ops
    
    @tf.function(jit_compile=True)
    def __call__(self, inp1, inp2):
        """
        Runs both operand order versions of the computation, compares their outputs and argmax results,
        and returns a dictionary containing both outputs and a boolean tensor indicating their closeness.

        This corresponds to fusing Model1 and Model2 from the issue, where:
        - Model1 does add = rev + mul
        - Model2 does add = mul + rev
        We check their differences after tf.math.argmax and also the add tensors.
        """
        # Common sub-computation
        mul = tf.matmul(inp2, inp1)   # shape inferred: (34, 18) * (18, 1) assuming the inputs shape
        rev = tf.reverse(mul, axis=[1])

        # Model1 computation (rev + mul)
        add1 = tf.add(rev, mul)
        argmax1 = tf.math.argmax(add1, axis=1)

        # Model2 computation (mul + rev)
        add2 = tf.add(mul, rev)
        argmax2 = tf.math.argmax(add2, axis=1)

        # Compare addition results (elementwise)
        add_close = tf.reduce_all(
            tf.abs(add1 - add2) <= 1e-3
        )  # boolean scalar - tolerance inferred from issue atol=0.001

        # Compare argmax results (elementwise)
        argmax_close = tf.reduce_all(
            tf.equal(argmax1, argmax2)
        )  # boolean scalar

        # Return all relevant results and combined comparison
        return {
            "add1": add1,
            "argmax1": argmax1,
            "add2": add2,
            "argmax2": argmax2,
            "add_close": add_close,
            "argmax_close": argmax_close,
            # Overall boolean whether model outputs fully match within tolerance
            "fully_match": tf.logical_and(add_close, argmax_close),
            # Also return difference tensors for debugging purposes
            "add_diff": add1 - add2,
        }

def my_model_function():
    # Return a fresh instance of MyModel; no weights to load or special initialization
    return MyModel()

def GetInput():
    # Based on example inputs in the issue and to avoid shape conflicts in matmul:
    # inp1 shape = [1, 18], inp2 shape = [34, 1]
    # matmul(inp2, inp1) => (34,1) x (1,18) = (34,18)
    # This is consistent with the supplied example and the use of reverse on axis=1 (dimension 18)
    inp1 = tf.random.uniform(shape=[1, 18], dtype=tf.float32)
    inp2 = tf.random.uniform(shape=[34, 1], dtype=tf.float32)
    return inp1, inp2

