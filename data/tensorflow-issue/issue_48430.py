# tf.random.uniform((4, 16), minval=0, maxval=50, dtype=tf.int64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create an embedding variable of shape (50, 16)
        self.embedding = tf.Variable(tf.random.uniform([50, 16]))

    def call(self, x):
        # Perform embedding lookup similarly to the original example
        return tf.nn.embedding_lookup(self.embedding, x)

def my_model_function():
    # Return an instance of MyModel with the embedding variable initialized
    return MyModel()

def GetInput():
    # Return a random integer tensor input choosing indices in [0,50)
    # Shape is (4,16) as per the original example in the issue
    return tf.random.uniform((4, 16), minval=0, maxval=50, dtype=tf.int64)

