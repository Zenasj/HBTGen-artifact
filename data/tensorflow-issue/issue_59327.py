# tf.ones([2, 3072], dtype=tf.float32)
import tensorflow as tf

SIZE = int(1024 * 3)  # 3072
RANGE = 80  # Number of Dense layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a sequence of Dense layers with output size SIZE
        self.layers_seq = [tf.keras.layers.Dense(SIZE) for _ in range(RANGE)]
        # Use Adam optimizer (as per original code)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def call(self, x, training=False):
        # Forward pass through the chain of Dense layers
        for layer in self.layers_seq:
            x = layer(x)
        return x

    @tf.function
    def train_step(self):
        # Runs one step of forward and backward passes with gradient update
        with tf.device("/gpu:0"):
            inp = tf.ones([2, SIZE], tf.float32)  # Input tensor shape as per original code
            with tf.GradientTape() as tape:
                y = self.call(inp, training=True)
            gradients = tape.gradient(y, self.trainable_weights)
            # Apply gradients to update weights
            self.opt.apply_gradients(zip(gradients, self.trainable_weights))
            return gradients

def my_model_function():
    # Return an instance of MyModel, optimizer and weights are initialized therein
    return MyModel()

def GetInput():
    # Return the input tensor matching the model's expected input shape and dtype
    return tf.ones([2, SIZE], tf.float32)

