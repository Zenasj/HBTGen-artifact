# tf.random.uniform((B, 2, H, W), dtype=tf.float32)  # Assuming channels_first Conv2D input with 2 inputs: range and doppler

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv block for "range" input
        self.c1_r = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c1_r")
        self.c2_r = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c2_r")
        self.m1_r = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first", name="m1_r")
        self.c3_r = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c3_r")
        self.c4_r = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c4_r")
        self.m2_r = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first", name="m2_r")
        self.flatten_r = tf.keras.layers.Flatten(name="flatten_r")

        # Conv block for "doppler" input
        self.c1_d = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c1_d")
        self.c2_d = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c2_d")
        self.m1_d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first", name="m1_d")
        self.c3_d = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c3_d")
        self.c4_d = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_first", name="c4_d")
        self.m2_d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first", name="m2_d")
        self.flatten_d = tf.keras.layers.Flatten(name="flatten_d")

        # Fully connected layers after concatenation
        self.concat = tf.keras.layers.Concatenate(axis=1, name="features")
        self.fc_2 = tf.keras.layers.Dense(128, activation="relu", use_bias=True,
                                          kernel_regularizer=tf.keras.regularizers.l2(0.01), name="fc_2")
        self.drop_2 = tf.keras.layers.Dropout(0.2)
        self.fc_3 = tf.keras.layers.Dense(64, activation="relu", use_bias=True,
                                          kernel_regularizer=tf.keras.regularizers.l2(0.01), name="fc_3")
        self.drop_3 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(5, use_bias=True, name="out")  # 5 classes output

    def call(self, inputs, training=False):
        # inputs expected as a tuple or list: (input_range, input_doppler)
        input_range, input_doppler = inputs

        # Range branch
        r = self.c1_r(input_range)
        r = self.c2_r(r)
        r = self.m1_r(r)
        r = self.c3_r(r)
        r = self.c4_r(r)
        r = self.m2_r(r)
        features_range = self.flatten_r(r)

        # Doppler branch
        d = self.c1_d(input_doppler)
        d = self.c2_d(d)
        d = self.m1_d(d)
        d = self.c3_d(d)
        d = self.c4_d(d)
        d = self.m2_d(d)
        features_doppler = self.flatten_d(d)

        # Concatenate features
        features = self.concat([features_range, features_doppler])

        x = self.fc_2(features)
        x = self.drop_2(x, training=training)
        x = self.fc_3(x)
        x = self.drop_3(x, training=training)
        out = self.out(x)

        return out


def my_model_function():
    # Return a compiled instance of MyModel.
    model = MyModel()
    # Compile with some placeholder loss and optimizer (user can recompile as needed)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def GetInput():
    # Based on Conv2D with channels_first and 2 inputs, input shape assumed:
    # Let's assume a batch of 1, with example input dimensions.
    # In the original code snippet, input shape variables were not given,
    # but by typical radar-like inputs it may be like (B, 2, 32, 32) or (B, 2, 64, 64).
    # We assume Example input shapes:
    #   input_range shape: (batch, channels=1, height=32, width=32)
    #   input_doppler shape: (batch, channels=1, height=32, width=32)
    # This matches channels_first convention.
    batch_size = 1
    height = 32
    width = 32
    channels = 1  # Each input has 1 channel

    # Create two random tensors as inputs for range and doppler respectively
    input_range = tf.random.uniform((batch_size, channels, height, width), dtype=tf.float32)
    input_doppler = tf.random.uniform((batch_size, channels, height, width), dtype=tf.float32)

    return (input_range, input_doppler)

