# tf.random.uniform((B=1, H=28, W=28, C=1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model architecture described:
        # Conv2D(16 filters, 3x3 kernel, stride 1, 'valid' padding, relu activation)
        self.conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), strides=(1, 1),
            padding='valid', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')
        
        # Note on Dropout difference between keras and tensorflow.keras:
        # keras.layers.Dropout(rate) accepts rate in [0,1], rate = fraction dropped
        # tensorflow.keras.layers.Dropout(rate) internally uses keep_prob = 1-rate
        # and expects keep_prob in (0, 1], so rate=1.0 causes ValueError.
        # To demonstrate, this model includes both Dropout versions as submodules.
        
        # keras Dropout: here just for comparison purposes (will not run in tf.keras)
        # but since we must implement a fused model, we simulate keras Dropout behavior.
        # We use a Lambda layer simulating keras Dropout (dropping fraction=rate).
        # Because keras Dropout rate=1.0 is valid (drops all neurons - disables layer),
        # we simulate with tf.keras.layers.Dropout(rate=rate_keras) but rate=0 disables dropout.
        # So for keras dropout with rate=1.0 (drops all), we simulate deterministic all-zero output.
        
        self.dropout_keras_rate = 1.0  # fraction dropped as in keras
        # We'll simulate keras dropout by a Lambda layer:
        # If rate=1.0, output zeros; else scale input by keep_prob.
        # (We assume deterministic behavior for demonstration.)
        self.dropout_keras = tf.keras.layers.Lambda(
            lambda x: tf.zeros_like(x) if self.dropout_keras_rate >= 1.0 else x * (1.0 - self.dropout_keras_rate),
            name='keras_dropout_sim')
        
        # tensorflow.keras Dropout layer (rate=fraction dropped),
        # but internally rate is converted to keep_prob=1-rate for dropout mask.
        # For demonstration, this Dropout will raise ValueError if rate=1.0,
        # so we use rate=0.5 here to avoid error.
        self.dropout_tf_rate = 0.5  # less than 1.0 to avoid ValueError
        self.dropout_tf = tf.keras.layers.Dropout(rate=self.dropout_tf_rate)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        
        # Apply simulated keras dropout (all zeros for rate=1.0)
        x_keras = self.dropout_keras(x)
        # Apply tensorflow.keras dropout (with rate=0.5)
        x_tf = self.dropout_tf(x, training=training)
        
        # Flatten both dropout outputs
        x_keras_flat = self.flatten(x_keras)
        x_tf_flat = self.flatten(x_tf)
        
        # Compute logits for each dropout variant
        out_keras = self.dense(x_keras_flat)
        out_tf = self.dense(x_tf_flat)
        
        # Compare outputs for demonstration: check numerical closeness
        # Because the keras dropout "drops all" (outputs zeros), result differs from tf dropout
        # Return a dictionary with outputs and comparison
        is_close = tf.reduce_all(tf.abs(out_keras - out_tf) < 1e-5)
        
        return {
            'output_keras_dropout': out_keras,
            'output_tf_dropout': out_tf,
            'outputs_close': is_close
        }

def my_model_function():
    # Return an instance of MyModel with the dropout differences illustrated.
    return MyModel()

def GetInput():
    # Return a random input tensor matching input_shape=(28, 28, 1)
    # with batch size 1 for simplicity.
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

