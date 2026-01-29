# tf.random.uniform((B, 4), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using tf.keras layers and optimizer as recommended to avoid the tensor conversion error.
        self.dense1 = tf.keras.layers.Dense(24, activation='relu', input_shape=(4,))
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(2, activation='linear')
        # Adam optimizer; learning rate set to 0.001 as in the original example
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Mean squared error loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
    
    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

def my_model_function():
    # Return an instance of MyModel with optimizer and loss configured
    return MyModel()

def GetInput():
    # Generate a random input matching shape (batch_size=1, 4 features), dtype float32
    # This matches the input_dim=4 from the original model
    return tf.random.uniform((1, 4), dtype=tf.float32)

