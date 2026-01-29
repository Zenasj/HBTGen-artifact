# tf.random.uniform((B, 12), dtype=tf.float32) ← Based on input to build_model() Dense layer input_shape=(12,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # some_model with an RNN inside, takes input shape (None, 10, 10)
        self.some_model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(64, input_shape=(10, 10)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])

        # model which uses a custom loss calling some_model internally
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(12,)),
            tf.keras.layers.Dense(10, activation='tanh')
        ])

        self.optimizer = tf.optimizers.Adam(0.001)

    def custom_loss(self, y_true, y_pred):
        # y_pred shape is (batch_size, 10) due to last Dense with 10 units
        # Expand dims and tile to shape (batch_size, 10, 10) to feed into some_model
        y_cred = tf.tile(tf.expand_dims(y_pred, axis=2), (1, 1, 10))
        # Call some_model on the tiled tensor
        some_model_out = self.some_model(y_cred)
        # Compute squared sum as the loss value
        loss = tf.reduce_sum(some_model_out ** 2)
        return loss

    @tf.function
    def call(self, inputs, training=None):
        # Forward pass through the primary model
        y_pred = self.model(inputs)
        return y_pred

    @tf.function
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.custom_loss(y_true, y_pred)
        grads = tape.gradient(loss, self.model.trainable_variables + self.some_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + self.some_model.trainable_variables))
        return {'loss': loss}

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Input shape expected by MyModel.model is (batch_size, 12)
    # Return random float tensor simulating a batch of 32 examples
    return tf.random.uniform((32, 12), dtype=tf.float32)

