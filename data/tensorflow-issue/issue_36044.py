# tf.random.uniform((B, BOARD_SIZE * 3), dtype=tf.float32)
import tensorflow as tf

# Assumption: BOARD_SIZE is the size of the tic-tac-toe board (likely 9 for 3x3)
# In original code BOARD_SIZE is used multiply by 3 for input and hidden layer sizes.
# We'll define BOARD_SIZE=9 as typical for tic-tac-toe.
BOARD_SIZE = 9

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build model layers per issue content.
        # Input shape is (BOARD_SIZE * 3,)
        # The model outputs two heads: 
        #   q_values: shape (BOARD_SIZE,)
        #   probabilities: softmax applied on q_values
        # Important fix: output order changed to have q_values as first output, probabilities second,
        # aligning losses with outputs to avoid training issues described in the issue.
        self.dense1 = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')
        self.dense2 = tf.keras.layers.Dense(BOARD_SIZE * 3 * 100, activation='relu')
        self.dense3 = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')
        self.q_values_layer = tf.keras.layers.Dense(BOARD_SIZE, activation=None, name='q_values')
        self.probabilities_layer = tf.keras.layers.Softmax(name='probabilities')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        q_values = self.q_values_layer(x)
        probabilities = self.probabilities_layer(q_values)
        # Return outputs as (q_values, probabilities) per final recommended working version
        return q_values, probabilities

def my_model_function():
    # Create, compile and return an instance of MyModel.
    model = MyModel()
    # Compile with losses aligned with output ordering:
    # q_values has MSE loss, probabilities have no loss (None).
    # This solves the original training issue described in the issue.
    # Note: experimental_run_tf_function argument no longer supported, so omitted.
    model.compile(
        optimizer='adam',
        loss=[tf.keras.losses.MeanSquaredError(), None]
    )
    return model

def GetInput():
    # Return a random input tensor shaped (batch_size, BOARD_SIZE * 3)
    # The batch size can be 5 as a nominal example.
    # Data type float32 for compatibility with Dense layers.
    batch_size = 5
    return tf.random.uniform((batch_size, BOARD_SIZE * 3), dtype=tf.float32)

