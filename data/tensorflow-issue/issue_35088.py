# tf.random.uniform((B, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend as K

class MyModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # Define the model architecture similar to mlp_model() from issue
        self.dense1 = layers.Dense(3, input_dim=3,
                                   kernel_initializer='glorot_uniform',
                                   activation='elu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(160,
                                   activation='elu',
                                   kernel_regularizer=regularizers.l2(0.001))
        self.gaussian_noise = layers.GaussianNoise(0.3)
        self.dense3 = layers.Dense(160,
                                   activation='elu',
                                   kernel_regularizer=regularizers.l2(0.003))
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense4 = layers.Dense(160,
                                   activation='elu',
                                   kernel_regularizer=regularizers.l2(0.003))
        self.dense5 = layers.Dense(160,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.004))
        self.out = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.gaussian_noise(x, training=training)
        x = self.dense3(x)
        x = self.dropout2(x, training=training)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.out(x)

    def predict_with_uncertainty(self, x, n_iter=100):
        """
        Perform Monte Carlo dropout predictions with dropout active during inference.

        Args:
          x: tf.Tensor input
          n_iter: int, number of stochastic forward passes

        Returns:
          Tuple (mean prediction, stddev prediction) tensors
        """
        predictions = []
        for _ in range(n_iter):
            # training=True makes Dropout/GaussianNoise active
            preds = self(x, training=True)
            predictions.append(preds)
        stacked = tf.stack(predictions, axis=0)  # shape: (n_iter, batch, 1)
        mean_pred = tf.reduce_mean(stacked, axis=0)
        std_pred = tf.math.reduce_std(stacked, axis=0)
        return mean_pred, std_pred


def my_model_function():
    # We return an instance of MyModel with default dropout rate.
    model = MyModel()
    # Compile similarly to original:
    # Loss function
    def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=root_mean_squared_error,
                  metrics=['mean_squared_error', 'mean_absolute_error', root_mean_squared_error])
    return model


def GetInput():
    # According to the example, input features are vectors of dimension 3,
    # batch size can be arbitrary, here we choose 10.
    batch_size = 10
    input_dim = 3
    # Generate a random float32 tensor with uniform values in range [0,1)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

