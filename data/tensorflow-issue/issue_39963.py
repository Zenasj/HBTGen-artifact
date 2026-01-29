# tf.random.uniform((B, 36, 1), dtype=tf.float32) ← inferred input shape from model input_shape=(36,1)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Rebuild the architecture described in the issue:
        # Sequential model with Conv1D, BatchNorm, Dropout, Flatten, Dense layers
        self.conv1 = Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(36,1))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.2)

        self.conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dropout3 = Dropout(0.5)

        self.dense2 = Dense(1, activation='sigmoid')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)

        output = self.dense2(x)
        return output

def my_model_function():
    # Return a new instance of MyModel
    # As the original code used Adam(lr=0.005), we follow common practice and set learning rate here for downstream use
    model = MyModel()
    # Model needs to be built/initialized for Keras weight variables; 
    # to mimic Sequential behavior, build model by calling once on dummy input
    dummy_input = tf.random.uniform((1, 36, 1), dtype=tf.float32)
    model(dummy_input, training=False)
    # Compile with same settings to match user's code
    model.compile(optimizer=Adam(learning_rate=0.005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random float32 tensor shaped (batch_size, 36, 1)
    # Batch size chosen moderately to simulate typical training batch; e.g., 32
    batch_size = 32
    return tf.random.uniform((batch_size, 36, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

