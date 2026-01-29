# tf.random.uniform((B=33, W=8), dtype=tf.float32) ← inferred from jnp.ones([33, 8]) input shape in the Flax example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # From the original Flax NN structure:
        # features = [[2,3,4],[2,3]]
        # kernel_sizes = [[(2,), (4,), (5,)], [(2,), (4,)]]
        #
        # We'll implement two rounds of 1D convolutions with ReLU activations,
        # flatten, and dense layer for each conv block.
        #
        # Since TF conv1d expects kernel_size as an integer,
        # we extract the first value of the tuple.
        #
        # We will follow roughly the logic:
        # For each boosting_round in [0,1]:
        #  For each depth in features[boosting_round]:
        #    Conv1D -> ReLU
        #  Flatten -> Dense(1)
        #
        # Sum outputs from both rounds as final output.

        # Round 0 conv and dense layers
        self.conv_0_0 = tf.keras.layers.Conv1D(filters=2, kernel_size=2, padding='same', name="conv_0_0")
        self.conv_0_1 = tf.keras.layers.Conv1D(filters=3, kernel_size=4, padding='same', name="conv_0_1")
        self.conv_0_2 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', name="conv_0_2")
        self.dense_0 = tf.keras.layers.Dense(1, name="dense_0")

        # Round 1 conv and dense layers
        self.conv_1_0 = tf.keras.layers.Conv1D(filters=2, kernel_size=2, padding='same', name="conv_1_0")
        self.conv_1_1 = tf.keras.layers.Conv1D(filters=3, kernel_size=4, padding='same', name="conv_1_1")
        self.dense_1 = tf.keras.layers.Dense(1, name="dense_1")

        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        # inputs shape expected: (batch_size, length, channels)
        # Original JAX input is (batch=33, width=8), no channel dimension explicitly given.
        # In TF conv1d, the input shape is (batch, length, channels).
        #
        # Since original input is (33, 8) -- looks like batch=33 and width=8,
        # but conv1d needs channels dimension.
        # We'll assume inputs shape: (batch_size=33, length=8, channels=1)
        # So we expect GetInput to add channel dimension correspondingly.

        # Round 0
        x0 = inputs
        x0 = self.conv_0_0(x0)
        x0 = self.act(x0)
        x0 = self.conv_0_1(x0)
        x0 = self.act(x0)
        x0 = self.conv_0_2(x0)
        x0 = self.act(x0)
        x0 = tf.reshape(x0, [tf.shape(x0)[0], -1])  # flatten
        out0 = self.dense_0(x0)  # (batch, 1)

        # Round 1
        x1 = inputs
        x1 = self.conv_1_0(x1)
        x1 = self.act(x1)
        x1 = self.conv_1_1(x1)
        x1 = self.act(x1)
        x1 = tf.reshape(x1, [tf.shape(x1)[0], -1])  # flatten
        out1 = self.dense_1(x1)

        # Sum outputs as in original: jnp.sum(y_rounds, axis=0)
        # y_rounds shape was (n_rounds, batch, 1)
        # So sum across rounds => sum of out0 and out1 (both shape (batch,1))
        output = out0 + out1  # shape (batch, 1)
        return output

def my_model_function():
    # Return an instance of MyModel with initialized weights by calling Build.
    # We use typical input shape [33, 8, 1] to build the model.
    model = MyModel()
    # We force build by calling model on a dummy input
    dummy_input = tf.zeros((33, 8, 1), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor matching the expected input shape:
    # batch=33, length=8, channels=1 (added channels dim for TF conv1d)
    return tf.random.uniform(shape=(33, 8, 1), dtype=tf.float32)

