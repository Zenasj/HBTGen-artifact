# tf.random.uniform((B, 10), dtype=tf.float32)  # Assuming input shape based on typical X.shape[1] as 10

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def normal_sp(params):
    # params shape: (..., 2)
    # Split parameters into loc and scale (stddev)
    loc, scale = tf.split(params, num_or_size_splits=2, axis=-1)
    # Ensure scale is positive via softplus
    scale = tf.nn.softplus(scale) + 1e-6
    return tfd.Normal(loc=loc, scale=scale)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # To replicate the reported structure, batch normalization and dropout in between Dense layers
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(128, activation='relu')
        self.params_layer = tf.keras.layers.Dense(2)
        self.dist_lambda = tfp.layers.DistributionLambda(normal_sp)

    def call(self, inputs, training=False):
        x = self.bn0(inputs, training=training)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.bn3(x, training=training)
        x = self.dense4(x)
        params = self.params_layer(x)
        dist = self.dist_lambda(params)
        return dist

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Based on original code, input shape is (batch_size, X.shape[1])
    # Since X.shape[1] is unknown, assuming 10 features
    batch_size = 4
    input_shape = (batch_size, 10)
    # Return a random float32 tensor as input
    return tf.random.uniform(input_shape, dtype=tf.float32)

