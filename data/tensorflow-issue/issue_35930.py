# tf.random.uniform((B,), dtype=tf.float32) ← The model's weights have shape (inp1,) and (inp2,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, inp1=100, inp2=200):
        super(MyModel, self).__init__()
        self.x1 = self.add_weight(name='w1', shape=[inp1], initializer='random_normal')
        self.x2 = self.add_weight(name='w2', shape=[inp2], initializer='random_normal')

    def call(self, x):
        # This model's call simply returns the input unchanged as per original code
        return x

def my_model_function():
    # Return an instance of MyModel with default input sizes 100 and 200
    return MyModel()

def GetInput():
    # According to original code, call accepts an arbitrary input and returns it.
    # The model does not specify concrete input shape requirements,
    # so return a random float32 tensor of shape (1,) as a minimal dummy input.
    # This shape works with the call method that returns input as is.
    return tf.random.uniform((1,), dtype=tf.float32)

