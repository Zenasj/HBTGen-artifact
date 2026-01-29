# tf.random.uniform((batch_size, 30, 7, 3208), dtype=tf.int32), tf.random.uniform((batch_size, 30, 3, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, losses
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.layers import LeakyReLU

# Since vocabsize was not given, we set a plausible default.
# Assumptions:
# - vocabsize is the size of the vocabulary used for embedding the "text" input.
# - batch_size is dynamic in GetInput.
vocabsize = 5000

def regDense(units):
    return layers.Dense(
        units,
        activation=LeakyReLU(),
        kernel_initializer=initializers.HeNormal(),
        kernel_regularizer=l2(0.001),
        bias_initializer=initializers.Zeros(),
        bias_regularizer=l1_l2(0.003, 0.02)
    )

def regLSTM(units):
    return layers.LSTM(
        units,
        kernel_regularizer=l1_l2(0.0001, 0.0003),
        kernel_initializer=initializers.GlorotNormal(),
        bias_initializer=initializers.Zeros(),
        bias_regularizer=l1_l2(0.0002, 0.002),
        return_sequences=True,
    )

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Inputs are expected:
        # num_inp shape: (30, 3, 1)
        # text_inp shape: (30, 7, 3208) - but this is tokenized integers, so embedding input shape uses integer dtype
        # embedding output dim = 152 as per original code
        self.vocabsize = vocabsize
        self.embedding_dim = 152

        # Embedding layer for text input
        self.embed = layers.Embedding(self.vocabsize, self.embedding_dim)

        # Stacked time distributed LSTMs for text input
        self.tlstm1 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))
        self.tdrop1 = layers.Dropout(0.2)

        self.tlstm2 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))
        self.tdrop2 = layers.Dropout(0.2)

        self.tlstm3 = layers.TimeDistributed(layers.TimeDistributed(regLSTM(256)))
        self.tdrop3 = layers.Dropout(0.2)

        # LSTMs for numerical input (30,3,1)
        self.nlstm1 = layers.TimeDistributed(regLSTM(256))
        self.ndrop1 = layers.Dropout(0.2)

        self.nlstm2 = layers.TimeDistributed(regLSTM(256))
        self.ndrop2 = layers.Dropout(0.2)

        # Dense layers on numeric side
        self.ndense1 = regDense(213)
        self.ndrop3 = layers.Dropout(0.5)

        self.ndense2 = regDense(170)
        self.ndrop4 = layers.Dropout(0.5)

        self.ndense3 = regDense(128)
        self.nrsp = layers.Reshape((30, 384))  # reshape to (30, 384)
        self.ndrop5 = layers.Dropout(0.5)

        # Dense layers on text side 
        self.tdense1 = regDense(200)
        self.tdrop4 = layers.Dropout(0.5)

        self.tdense2 = regDense(144)

        # Pooling layers on text side 
        self.tpool2 = layers.MaxPooling3D((1, 1, 401), padding='same')
        self.trsp1 = layers.Reshape((30, 1152, 7))
        self.tpool3 = layers.MaxPooling2D((1, 3), padding='same')
        self.trsp2 = layers.Reshape((2688, 30))
        self.tpool4 = layers.MaxPooling1D(7, padding='same')
        self.trsp3 = layers.Reshape((30, 384))
        self.tdrop5 = layers.Dropout(0.5)

        # Concatenate and further processing dense layers
        self.concat_layer = layers.Concatenate()
        self.prc1 = regDense(384)
        self.pdrop1 = layers.Dropout(0.5)
        self.prc2 = regDense(192)
        self.pdrop2 = layers.Dropout(0.5)
        self.prc3 = regDense(96)
        self.pdrop3 = layers.Dropout(0.5)
        self.prc4 = regDense(48)
        self.pdrop4 = layers.Dropout(0.5)
        self.prc5 = regDense(24)
        self.pdrop5 = layers.Dropout(0.3)
        self.prc6 = regDense(12)

        # Final output dense + global avg pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.last_dense = layers.Dense(
            1,
            activation='relu',
            kernel_regularizer=l2(0.0008),
            kernel_initializer=initializers.HeNormal(),
            bias_initializer=initializers.Zeros(),
            bias_regularizer=l1_l2(0.003, 0.02),
            name='output'
        )

    def call(self, inputs, training=False):
        # inputs is a tuple/list of two tensors: (num_inp, text_inp)
        num_inp, text_inp = inputs

        # Text branch
        x = self.embed(text_inp)  # shape: (batch, 30, 7, 3208) → (batch, 30, 7, 152)
        x = self.tlstm1(x)
        x = self.tdrop1(x, training=training)
        x = self.tlstm2(x)
        x = self.tdrop2(x, training=training)
        x = self.tlstm3(x)
        x = self.tdrop3(x, training=training)
        x = self.tdense1(x)
        x = self.tdrop4(x, training=training)
        x = self.tdense2(x)
        x = self.tpool2(x)
        x = self.trsp1(x)
        x = self.tpool3(x)
        x = self.trsp2(x)
        x = self.tpool4(x)
        x = self.trsp3(x)
        x = self.tdrop5(x, training=training)

        # Numeric branch
        y = self.nlstm1(num_inp)
        y = self.ndrop1(y, training=training)
        y = self.nlstm2(y)
        y = self.ndrop2(y, training=training)
        y = self.ndense1(y)
        y = self.ndrop3(y, training=training)
        y = self.ndense2(y)
        y = self.ndrop4(y, training=training)
        y = self.ndense3(y)
        y = self.nrsp(y)
        y = self.ndrop5(y, training=training)

        # Concatenate branches
        combined = self.concat_layer([x, y])
        combined = self.prc1(combined)
        combined = self.pdrop1(combined, training=training)
        combined = self.prc2(combined)
        combined = self.pdrop2(combined, training=training)
        combined = self.prc3(combined)
        combined = self.pdrop3(combined, training=training)
        combined = self.prc4(combined)
        combined = self.pdrop4(combined, training=training)
        combined = self.prc5(combined)
        combined = self.pdrop5(combined, training=training)
        combined = self.prc6(combined)
        pooled = self.global_avg_pool(combined)

        output = self.last_dense(pooled)

        return output

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile with Adam optimizer, MSE loss and accuracy metric as used in original code
    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
    return model

def GetInput():
    # Produce a random input tuple (num_inp, text_inp) matching expected input shapes and types

    # For numeric input: shape (batch_size, 30, 3, 1), dtype float32
    num_inp = tf.random.uniform(shape=(1, 30, 3, 1), dtype=tf.float32)

    # For text input: shape (batch_size, 30, 7, 3208), dtype int32, values in vocab range [0, vocabsize)
    # The original code receives embed input for text_inp, so input must be integer tokens
    text_inp = tf.random.uniform(shape=(1, 30, 7, 3208), minval=0, maxval=vocabsize, dtype=tf.int32)

    return (num_inp, text_inp)

