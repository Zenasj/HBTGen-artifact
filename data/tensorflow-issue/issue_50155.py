# tf.random.uniform((B, 100, 1), dtype=tf.float32)  ← Input shape is (batch_size, 100, 1)

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, GlobalMaxPooling1D, Concatenate, BatchNormalization, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Initial Conv layer with 20 filters, kernel size 1
        self.C0 = Conv1D(20, kernel_size=1, strides=1, activation='relu', name='C0')

        # Parallel Conv layers after initial conv
        self.C1 = Conv1D(50, kernel_size=2, strides=1, activation='relu', name='C1')
        self.DR1 = Dropout(0.3, name='DR1')
        self.MP1 = GlobalMaxPooling1D(name='MP1')

        self.C2 = Conv1D(35, kernel_size=3, strides=1, activation='relu', name='C2')
        self.DR2 = Dropout(0.3, name='DR2')
        self.MP2 = GlobalMaxPooling1D(name='MP2')

        self.C3 = Conv1D(25, kernel_size=4, strides=1, activation='relu', name='C3')
        self.DR3 = Dropout(0.3, name='DR3')
        self.MP3 = GlobalMaxPooling1D(name='MP3')

        self.C4 = Conv1D(20, kernel_size=5, strides=1, activation='relu', name='C4')
        self.DR4 = Dropout(0.3, name='DR4')
        self.MP4 = GlobalMaxPooling1D(name='MP4')

        self.concat = Concatenate(axis=1, name='concat')
        self.BN = BatchNormalization(name='BN')
        self.output_layer = Dense(units=1, activation='linear', name='output')

    def call(self, inputs, training=False):
        # inputs: (batch_size, 100, 1)
        x0 = self.C0(inputs)  # (batch_size, 100, 20)

        x1 = self.C1(x0)      # (batch_size, ?, 50) - depends on kernel size and strides (valid padding default)
        x1 = self.DR1(x1, training=training)
        x1 = self.MP1(x1)     # (batch_size, 50)

        x2 = self.C2(x0)      # (batch_size, ?, 35)
        x2 = self.DR2(x2, training=training)
        x2 = self.MP2(x2)     # (batch_size, 35)

        x3 = self.C3(x0)      # (batch_size, ?, 25)
        x3 = self.DR3(x3, training=training)
        x3 = self.MP3(x3)     # (batch_size, 25)

        x4 = self.C4(x0)      # (batch_size, ?, 20)
        x4 = self.DR4(x4, training=training)
        x4 = self.MP4(x4)     # (batch_size, 20)

        concat_out = self.concat([x1, x2, x3, x4])  # Along features axis: (batch_size, 50+35+25+20=130)
        bn_out = self.BN(concat_out, training=training)

        output = self.output_layer(bn_out)  # (batch_size, 1)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random float32 tensor with batch size 16 and shape (100,1)
    batch_size = 16
    return tf.random.uniform(shape=(batch_size, 100, 1), dtype=tf.float32)

