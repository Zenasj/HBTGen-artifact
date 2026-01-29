# tf.random.uniform((BATCH_SIZE, 40, 4), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras

window_length = 40
feats = 4
BATCH_SIZE_PER_REPLICA = 4096  # original large per-replica batch size (could be tuned)
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  # Global batch size across 8 GPUs

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder LSTM layers
        self.encoder_1 = keras.layers.LSTM(
            64, kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_1',
            batch_input_shape=(None, window_length, feats))
        self.dropout_1 = keras.layers.Dropout(0.25)
        self.encoder_2 = keras.layers.LSTM(
            32, kernel_initializer='he_uniform',
            return_sequences=True, name='encoder_2')
        self.dropout_2 = keras.layers.Dropout(0.25)
        self.encoder_3 = keras.layers.LSTM(
            16, kernel_initializer='he_uniform',
            return_sequences=False, name='encoder_3')
        self.dropout_3 = keras.layers.Dropout(0.25)

        # Bridge: RepeatVector to prepare for decoder
        self.bridge = keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge')

        # Decoder LSTM layers
        self.decoder_1 = keras.layers.LSTM(
            16, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_1')
        self.dropout_4 = keras.layers.Dropout(0.25)
        self.decoder_2 = keras.layers.LSTM(
            32, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_2')
        self.dropout_5 = keras.layers.Dropout(0.25)
        self.decoder_3 = keras.layers.LSTM(
            64, kernel_initializer='he_uniform',
            return_sequences=True, name='decoder_3')
        self.dropout_6 = keras.layers.Dropout(0.25)

        # Output layer: TimeDistributed dense matching feats size
        self.time_dist = keras.layers.TimeDistributed(
            keras.layers.Dense(feats), name='time_distributed')

    def call(self, inputs, training=False):
        # Forward pass through encoder
        x = self.encoder_1(inputs, training=training)
        x = self.dropout_1(x, training=training)
        x = self.encoder_2(x, training=training)
        x = self.dropout_2(x, training=training)
        x = self.encoder_3(x, training=training)
        x = self.dropout_3(x, training=training)

        # Repeat vector to prepare decoder input
        x = self.bridge(x)

        # Forward pass through decoder
        x = self.decoder_1(x, training=training)
        x = self.dropout_4(x, training=training)
        x = self.decoder_2(x, training=training)
        x = self.dropout_5(x, training=training)
        x = self.decoder_3(x, training=training)
        x = self.dropout_6(x, training=training)

        # Output reconstruction of original features at each timestep
        output = self.time_dist(x)
        return output

def my_model_function():
    """
    Create and compile the model under the MirroredStrategy scope.
    """
    with strategy.scope():
        model = MyModel()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='mse')
    return model

def GetInput():
    """
    Generate a random input tensor compatible with MyModel's expected input shape.
    The input shape is (BATCH_SIZE, window_length=40, feats=4).
    Note: batch size here matches the global batch size distributed across GPUs.
    """
    # Using dtype float32 as per TF default and model data expectations
    return tf.random.uniform(
        shape=(BATCH_SIZE, window_length, feats), dtype=tf.float32)

# Reasoning / Notes from the issue and thread:
# - The original problem arises because the last batch is smaller than the batch size,
#   and this triggers errors with MirroredStrategy distributed training.
# - The recommended workaround is to use drop_remainder=True in the batch call of the dataset pipeline,
#   to ensure all batches have the exact same batch size expected by the distribution strategy.
# - Without drop_remainder=True, the last batch is smaller and shape mismatch causes slicing errors.
# - The callable model here reflects the Sequential model of the user issue, recreated as a subclassed Model.
# - Input generation reflects the distributed batch size (batch per replica * replicas).
# - The model design matches exactly the stack of LSTM layers, dropouts, and TimeDistributed output layer.
# - The strategy scope and compilation mirrors the user's approach to multi-GPU training.
# - No flatten or window ops are included here because they relate to dataset pipelines, not the model itself.

# The returned callable can be used with:
# model = my_model_function()
# output = model(GetInput())

# This code structure follows TensorFlow 2.x idioms and should be compatible with TF 2.20.0 XLA compilation.

