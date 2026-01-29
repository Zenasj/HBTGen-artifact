# tf.random.uniform((None, 10), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We model the old tf-1.2.1 Keras Sequential with a single Embedding layer,
        # input shape [None, 10], input dtype int32.
        self.embedding = tf.keras.layers.Embedding(
            input_dim=1000, 
            output_dim=64, 
            input_length=10,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),  # Old tf1.2.1 initializer had maxval=None; interpret as uniform random
            dtype=tf.int32,
            name="embedding_1"
        )
        
    def call(self, inputs):
        # inputs assumed int32 tensor of shape [batch_size, 10]
        return self.embedding(inputs)

def my_model_function():
    # Return an instance of MyModel, initialized as per old tf-1.2.1 config.
    return MyModel()

def GetInput():
    # Return valid input tensor matching model input: shape (batch_size, 10), dtype int32,
    # values in [0, 999] as input_dim=1000
    batch_size = 4  # arbitrary batch size
    input_length = 10
    return tf.random.uniform(
        shape=(batch_size, input_length), minval=0, maxval=1000, dtype=tf.int32
    )

