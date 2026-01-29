# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape from the example with keras.Input(shape=(784,))

import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    """
    Example of a custom loss class to replace the simple custom_loss function.
    This approach ensures proper serialization and deserialization.
    """
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
    def get_config(self):
        base_config = super().get_config()
        return base_config

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers just like in the issue example
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.predictions = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.predictions(x)

def my_model_function():
    """
    Return an instance of MyModel compiled with the custom loss and RMSprop optimizer,
    reflecting the original model setup.
    """
    model = MyModel()
    # Build model by calling with dummy input to initialize weights
    model(tf.zeros((1, 784)))
    model.compile(loss=CustomLoss(), optimizer=tf.keras.optimizers.RMSprop())
    return model

def GetInput():
    """
    Return a random tensor input matching the model's expected input shape (batch, 784).
    Using float32 dtype as typical for such inputs.
    """
    batch_size = 1  # batch size can be dynamic, 1 for testing here
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

