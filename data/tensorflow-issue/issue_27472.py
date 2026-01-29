# tf.random.uniform((BATCH_SIZE, None, 1), dtype=tf.float32) ← input shape based on (None,1) time series input with variable length sequences
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the architecture from the issue:
        # Input: (None, 1) variable length sequences of feature dim 1
        # Layers: SimpleRNN(100 units), Dense(100 units), Dense(1 unit)
        self.rnn = tf.keras.layers.SimpleRNN(100, name='RNN')
        self.hidden = tf.keras.layers.Dense(100, name='hidden')
        self.target = tf.keras.layers.Dense(1, name='target')

    def call(self, inputs, training=False):
        """
        inputs: tensor of shape (batch_size, time_steps, 1)
        Returns: tensor of shape (batch_size, 1) - logits or predictions
        """
        x = self.rnn(inputs)
        x = self.hidden(x)
        x = self.target(x)
        return x

def my_model_function():
    # Return an instance of MyModel. 
    model = MyModel()

    # Compile the model to match the issue example:
    # optimizer='SGD', loss='binary_crossentropy'
    # run_eagerly can be passed if desired, but omitted here (default is False).
    model.compile(optimizer='SGD', loss='binary_crossentropy')
    return model

def GetInput():
    # Provide a random input tensor consistent with the input shape:
    # The model input was defined as (None, 1), i.e. variable timesteps with 1 feature.
    # The batch size is BATCH_SIZE = 10 in the example.
    # Time steps are variable; choosing fixed length 10 for this input generation.
    BATCH_SIZE = 10
    TIME_STEPS = 10
    FEATURES = 1
    return tf.random.uniform((BATCH_SIZE, TIME_STEPS, FEATURES), dtype=tf.float32)

