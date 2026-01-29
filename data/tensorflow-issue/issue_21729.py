# tf.random.uniform((B, 2), dtype=tf.float32), multi-input tuple of two tensors each shape (B, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Two example approaches fused into one model:
        # 1) Model using tuple input of two tensors added before Dense+softmax
        self.dense_softmax = tf.keras.layers.Dense(2, activation="softmax")
        # 2) Model using dict inputs processed by separate Dense layers then combined and projected
        self.dense1 = tf.keras.layers.Dense(2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=True, mask=None):
        # The inputs args can be either:
        # - tuple of two tensors (as in the first example)
        # - dict with keys 'x1' and 'x2' (as in the second example)
        # Implement logic to identify input type and compute accordingly.

        if isinstance(inputs, tuple) or isinstance(inputs, list):
            # Expecting a tuple/list of two tensors
            x = inputs[0] + inputs[1]
            result1 = self.dense_softmax(x)  # Shape (batch, 2)
            return result1
        elif isinstance(inputs, dict):
            # Inputs is a dict with keys 'x1' and 'x2', each tensor shape (batch, feature)
            x1 = inputs.get('x1')
            x2 = inputs.get('x2')
            if x1 is None or x2 is None:
                raise ValueError("Input dict must contain keys 'x1' and 'x2'")
            x1_out = self.dense1(x1)
            x2_out = self.dense2(x2)
            combined = x1_out + x2_out
            result2 = self.dense3(combined)  # Shape (batch, 1)
            return result2
        else:
            raise ValueError("Unsupported input type for MyModel: must be tuple/list or dict")

def my_model_function():
    # Return an instance of MyModel, no pretrained weights
    return MyModel()

def GetInput():
    # Provide a sample input compatible with MyModel()
    # Since MyModel handles two input forms: tuple of tensors or dict of tensors,
    # we will return a tuple of two tensors of shape (batch=128, 2) with float32,
    # matching the original example from the issue.

    batch_size = 128
    shape = (batch_size, 2)
    x1 = tf.random.uniform(shape, dtype=tf.float32)
    x2 = tf.random.uniform(shape, dtype=tf.float32)
    # Returning a tuple, consistent with the first example that caused validation errors
    return (x1, x2)

