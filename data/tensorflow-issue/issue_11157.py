# tf.random.uniform((10, 20, 4), dtype=tf.float32) ← inferred input shape from example LSTM batch_input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # From the minimal reproducible snippet:
        # LSTM with 20 units, output sequences, batch_input_shape=(10,20,4)
        # followed by Dense layer to 3 classes with softmax activation.
        # This recreates the example architecture which encountered the pickling issue.
        self.lstm = tf.keras.layers.LSTM(
            20,
            return_sequences=True,
            stateful=False,
            # batch_input_shape fixed for model building; input shape excludes batch size:
            # batch_input_shape was (10,20,4) so input shape is (20,4)
            input_shape=(20, 4)
        )
        self.dense = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        return self.dense(x)

def my_model_function():
    # Build and return an instance of MyModel
    model = MyModel()
    # Build the model by calling it once with dummy input
    dummy_input = GetInput()
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor with shape (10,20,4) matching batch_input_shape in original example
    # dtype float32 as typical for TF inputs
    return tf.random.uniform((10, 20, 4), dtype=tf.float32)


# Note on context / reasoning:
# The original GitHub issue revolves around a TypeError: "can't pickle _thread.lock objects".
# The error occurs when trying to pickle (deepcopy) models with TensorFlow 1.2.1 + Keras in a multiprocessing or sklearn wrapper context.
# The minimal repro provided is a Keras Sequential model with one LSTM layer followed by Dense.
# Attempting to pickle the entire Keras model leads to TypeError on _thread.lock objects internally.
#
# This code captures the minimal example as a subclassed tf.keras.Model (named MyModel as requested) with the same architecture.
# The GetInput function provides an input tensor shape compatible with the LSTM's batch input.
#
# This code is compatible with TensorFlow 2.20.0 as requested and can be compiled with XLA jit_compile if desired.
#
# This reformulation avoids the serialization/deserialization pitfalls by not serializing directly,
# and provides a clear, runnable minimal model and input for testing or debugging.
#
# The original issue's heart was about pickle and deepcopy on TF/Keras graphs and layers including _thread.lock objects.
# This standalone model encapsulation is a recommended pattern to avoid those pickling issues when using multiprocessing or sklearn wrappers.

