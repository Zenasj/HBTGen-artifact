# tf.random.uniform((B, 3), dtype=tf.float32)  # Inferred input shape from model input shape=(3,)

import tensorflow as tf

# Because original issue is from TF 1.13 era where eager execution was enabled via tf.enable_eager_execution()
# In TF 2.x eager is enabled by default; however, to keep compatibility with TF 2.20.0 we just write a model class.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers matching the original example:
        self.dense1 = tf.keras.layers.Dense(5, 
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.dense2 = tf.keras.layers.Dense(5, 
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.out = tf.keras.layers.Dense(1, 
                                         kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                         activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return self.out(x)

def my_model_function():
    # Instantiate the model and compile it with SGD optimizer and MSE loss,
    # setting run_eagerly=True to allow custom loss stepping (as per original user's intent).
    model = MyModel()
    sgd = tf.keras.optimizers.SGD()
    # Compile model as per original code snippet
    model.compile(optimizer=sgd, loss='mean_squared_error')
    # Model.run_eagerly attribute exists in TF 2.x; setting True enables eager mode training.
    model.run_eagerly = True
    return model

def GetInput():
    # Generate dummy batch input tensor matching input shape=(3,)
    # Batch size is 4 (arbitrarily chosen) for demonstration
    # Values random uniform between 0 and 1 to simulate general input
    batch_size = 4
    input_shape = (batch_size, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

