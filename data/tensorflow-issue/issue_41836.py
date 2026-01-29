# tf.random.uniform(()) ← The example input is a scalar (e.g., 1.5), so input shape is () (a scalar tensor)

import tensorflow as tf

# We cannot use numba or numpy functions with .numpy() calls inside a TensorFlow graph
# because those break tf.function and XLA compilation.
# So we need to reimplement func and gradfunc purely with TensorFlow ops.

@tf.custom_gradient
def func_tf(param, input):
    # Forward computation: param * input^2
    output = param * tf.square(input)
    
    def grad(dy):
        # Gradient w.r.t param is input^2 * dy, gradient w.r.t input is 2 * param * input * dy
        grad_param = tf.reduce_sum(dy * tf.square(input))  # Sum over batch if needed
        grad_input = dy * 2.0 * param * input
        return grad_param, grad_input
    
    return output, grad


class myLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        # Initialize the scalar weight param
        self.param = self.add_weight(name="param", shape=(), initializer="random_normal")
        
    def call(self, input):
        return func_tf(self.param, input)
    

class MyModel(tf.keras.Model):
    def __init__(self, num_layers):
        super().__init__(name='')
        self._layers = [myLayer() for _ in range(num_layers)]
        
    def call(self, input_tensor):
        x = input_tensor
        for layer in self._layers:
            x = layer(x)
        return x


def my_model_function():
    # Return an instance of MyModel with 3 layers as in the original example
    return MyModel(3)


def GetInput():
    # Input is a scalar tensor, so produce a single random float scalar tensor
    return tf.random.uniform(shape=(), dtype=tf.float32)

