# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred as (batch_size, 2) matching the input_5 layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The architecture is reconstructed from the Keras JSON model configuration

        # Reshape layers from the input of shape (None, 2) to various shapes
        self.reshape_2 = tf.keras.layers.Reshape(target_shape=(1, 2), name="reshape_2")
        self.reshape_1 = tf.keras.layers.Reshape(target_shape=(1, 2), name="reshape_1")
        self.reshape = tf.keras.layers.Reshape(target_shape=(2, 1), name="reshape")
        self.reshape_5 = tf.keras.layers.Reshape(target_shape=(2, 1), name="reshape_5")
        self.reshape_3 = tf.keras.layers.Reshape(target_shape=(2, 1), name="reshape_3")
        self.reshape_4 = tf.keras.layers.Reshape(target_shape=(1, 1), name="reshape_4")

        # UpSampling1D layers with sizes as per config
        self.up_sampling1d_2 = tf.keras.layers.UpSampling1D(size=3, name="up_sampling1d_2")
        self.up_sampling1d_1 = tf.keras.layers.UpSampling1D(size=3, name="up_sampling1d_1")
        self.up_sampling1d = tf.keras.layers.UpSampling1D(size=2, name="up_sampling1d")

        # Zero padding
        self.zero_padding1d_1 = tf.keras.layers.ZeroPadding1D(padding=(0, 1), name="zero_padding1d_1")
        self.zero_padding1d = tf.keras.layers.ZeroPadding1D(padding=(0, 1), name="zero_padding1d")

        # Conv1D layers (filters and kernel size as per config)
        self.conv1d = tf.keras.layers.Conv1D(filters=2, kernel_size=1, padding="valid", activation="linear", name="conv1d")
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding="valid", activation="linear", name="conv1d_1")

        # MaxPooling1D layer
        self.max_pooling1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="max_pooling1d")

        # Dot layers with axes = -1 (last axis), normalize=False
        self.dot = tf.keras.layers.Dot(axes=-1, normalize=False, name="dot")
        self.dot_1 = tf.keras.layers.Dot(axes=-1, normalize=False, name="dot_1")

        # SpatialDropout1D and AlphaDropout
        self.spatial_dropout1d = tf.keras.layers.SpatialDropout1D(rate=0.4, name="spatial_dropout1d")
        self.alpha_dropout = tf.keras.layers.AlphaDropout(rate=0.24308844217362846, name="alpha_dropout")

        # Dropout layer with rate 0.4
        self.dropout = tf.keras.layers.Dropout(rate=0.4, name="dropout")

        # GlobalAveragePooling1D layers
        self.global_average_pooling1d = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pooling1d")
        self.global_average_pooling1d_1 = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pooling1d_1")

        # Flatten layers
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.flatten_1 = tf.keras.layers.Flatten(name="flatten_1")

        # Concatenate layer along last axis (axis=-1)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1, name="concatenate")

        # Output Dense layer with 2 units and ReLU activation
        self.dense_4 = tf.keras.layers.Dense(units=2, activation="relu", name="dense_4")

    def call(self, inputs, training=False):
        # The input shape is (batch_size, 2), dtype float32 as per config
        
        # Multiple reshapes from input
        r2 = self.reshape_2(inputs)      # (B, 1, 2)
        r1 = self.reshape_1(inputs)      # (B, 1, 2)
        r = self.reshape(inputs)         # (B, 2, 1)
        r5 = self.reshape_5(inputs)      # (B, 2, 1)
        r3 = self.reshape_3(inputs)      # (B, 2, 1)

        # UpSampling on reshaped inputs
        up2 = self.up_sampling1d_2(r2)   # (B, 3, 2)
        up1 = self.up_sampling1d_1(r1)   # (B, 3, 2)
        up = self.up_sampling1d(r)       # (B, 4, 1)

        # Zero padding the upsampled outputs for conv1d_1 and dot
        zp1 = self.zero_padding1d_1(up2) # (B, 4, 2)
        zp = self.zero_padding1d(up1)    # (B, 4, 2)

        # Conv1D layers applied
        c = self.conv1d(up)              # (B, 4, 2)
        c1 = self.conv1d_1(zp1)          # (B, 4, 4)

        # Dot product layer
        d = self.dot([zp, c])            # (B, 4, 4)

        # MaxPooling + SpatialDropout + AlphaDropout layers
        mp = self.max_pooling1d(r5)      # (B, 1, 1)
        sd = self.spatial_dropout1d(mp, training=training)  # (B, 1, 1)
        ad = self.alpha_dropout(sd, training=training)      # (B, 1, 1)

        # GlobalAveragePooling1D layers + reshape_4
        gap = self.global_average_pooling1d(r3)             # (B, 1)
        r4 = self.reshape_4(gap)                             # (B, 1, 1)
        gap1 = self.global_average_pooling1d_1(r4)          # (B, 1)

        # Dot_1 layer between conv1d_1 output and previous dot output
        d1 = self.dot_1([c1, d])              # (B, 4, 4)

        # Dropout and flatten layers
        do = self.dropout(d1, training=training)            # (B, 4, 4)
        flt = self.flatten(ad)                               # (B, 1)
        flt_1 = self.flatten_1(do)                           # (B, 16)

        # Concatenate inputs
        concat = self.concatenate([inputs, flt, gap1, flt_1])  # (B, 2 + 1 + 1 + 16 = 20)

        # Dense output layer
        out = self.dense_4(concat)                           # (B, 2)

        return out


def my_model_function():
    # Return an instance of MyModel with layers initialized as above
    model = MyModel()
    # Build the model once with a dummy input to create weights properly
    model(tf.zeros([1, 2], dtype=tf.float32))
    return model


def GetInput():
    # Return a random tensor input (batch size 4) of shape (4, 2), dtype float32
    # This matches the input shape expected by the model's InputLayer
    return tf.random.uniform(shape=(4, 2), dtype=tf.float32)

