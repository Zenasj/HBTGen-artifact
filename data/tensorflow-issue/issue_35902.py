# tf.random.uniform((1, 2, 2), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model wraps tf.keras.layers.Attention with a custom call signature:
      call(q, v, k, mask_q=None, mask_v=None, **kwargs)
    It demonstrates how to handle keyword arguments properly when calling the model 
    via the functional __call__ interface, avoiding issues with argument mismatch.

    Key points / assumptions:
    - Inputs q, v, k have shape (batch, seq_len, features), here assumed (1, 2, 2) for sample.
    - Masks mask_q and mask_v are optional, can be None.
    - Extra keyword arguments are accepted and ignored for forward pass,
      but can be logged or used internally.
    - The output is the attention result tensor of shape (batch, query_seq_len, features).
    """
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.attention = tf.keras.layers.Attention(use_scale=True)

    def call(self, q, v, k, mask_q=None, mask_v=None, **kwargs):
        # Optional: print or log extra kwargs to ensure they're caught if passed
        for key, value in kwargs.items():
            tf.print(f"[MyModel.call] Extra kwarg: {key} =", value)
        # Use the Attention layer with inputs and masks as a list
        return self.attention(inputs=[q, v, k], mask=[mask_q, mask_v])

def my_model_function():
    # Returns an instance of MyModel as required
    return MyModel()

def GetInput():
    # Returns a tuple of inputs (q, v, k) each with shape (1, 2, 2)
    # Random float32 tensors in [0,1) to simulate typical input features
    input_shape = (1, 2, 2)
    q = tf.random.uniform(input_shape, dtype=tf.float32)
    v = tf.random.uniform(input_shape, dtype=tf.float32)
    k = tf.random.uniform(input_shape, dtype=tf.float32)
    return (q, v, k)

