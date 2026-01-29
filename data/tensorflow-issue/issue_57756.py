# tf.random.uniform((32, 4, 16), dtype=tf.float32) ← inferred input shape from example x input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        # x is expected to be a list/tuple where x[0] has shape (32,4,16)
        return self.out(x[0])

class MyModelWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mem = None
        self.loss_mem = None

    def loss(self, x):
        # Simple example loss: linear function on output tensor
        return 2 * x - 1

    @tf.function
    def gradient(self, x):
        # Store intermediate outputs and losses - can be disabled if unused, but kept for illustrative consistency
        self.mem = tf.TensorArray(tf.float32, size=10)
        self.loss_mem = tf.TensorArray(tf.float32, size=10)
        for i in tf.range(10):
            out_i = self.model.call(x)
            loss_i = self.loss(out_i)
            self.mem = self.mem.write(i, out_i)
            self.loss_mem = self.loss_mem.write(i, loss_i)

        # Initialize gradient accumulation as a python list of zero tensors matching trainable variables
        ghat_old = [tf.zeros_like(v, dtype=tf.float32) for v in self.model.trainable_variables]

        # Reverse loop to accumulate gradients
        for i in tf.reverse(tf.range(10), [0]):
            # Create new list reference for gradients accumulation each iteration to avoid Tensor out-of-scope error
            ghat = [g for g in ghat_old]
            with tf.GradientTape() as g:
                g.watch(x)
                out_i = self.model.call(x)
            with tf.GradientTape() as g_loss:
                g_loss.watch(out_i)
                loss_i = self.loss(out_i)
            output_grad = g_loss.gradient(loss_i, out_i)
            ghat_update = g.gradient(
                out_i, self.model.trainable_variables, output_gradients=output_grad
            )
            # Accumulate gradients manually
            for j in range(len(ghat)):
                ghat[j] = ghat[j] + ghat_update[j]
            ghat_old = ghat
        return ghat

# Provide unified class naming as required
MyModel = MyModelWrapper

def my_model_function():
    # Initialize Inner model and wrap in Outer (renamed MyModelWrapper -> MyModel)
    inner_model = MyModel()
    return MyModel(inner_model.model)

def GetInput():
    # According to original example, input is a list[x0, x1]
    # x0 shape: (32, 4, 16), x1 shape: (64, 10); but the model uses only x[0]
    x0 = tf.random.uniform((32, 4, 16), dtype=tf.float32)
    x1 = tf.random.uniform((64, 10), dtype=tf.float32)
    return [x0, x1]

