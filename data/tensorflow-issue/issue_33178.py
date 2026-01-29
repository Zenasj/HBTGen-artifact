# tf.random.uniform((256, 100, 9, 11, 1), dtype=tf.float32) ← inferred input shape (batch=256, time=100, height=9, width=11, channels=1)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Dropout, TimeDistributed

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='RCNN')
        # Model hyperparameters 
        self.n_filters = 32
        self.n_fc = 256
        self.n_output = 3  # multiclass classification as per one-hot labels in original code
        self.N_BATCH = 256
        self.N_NODES = 256
        self.DROPOUT = 0.5

        self.out_activation = "softmax"

        # 3 Conv2D layers
        self.conv1 = Conv2D(filters=self.n_filters, strides=1, padding='same', 
                            activation='tanh', kernel_size=3)
        self.conv2 = Conv2D(filters=self.n_filters*2, strides=1, padding='same', 
                            activation='tanh', kernel_size=3)
        self.conv3 = Conv2D(filters=self.n_filters*4, strides=1, padding='same', 
                            activation='tanh', kernel_size=3)

        # Dense and dropout after flattening conv features
        self.dense1 = Dense(self.n_fc)
        self.dropout1 = Dropout(self.DROPOUT)

        # Two stacked LSTMs with return_sequences=True for TimeDistributed output
        self.lstm1 = LSTM(self.N_NODES, recurrent_initializer='orthogonal', return_sequences=True)
        self.lstm2 = LSTM(self.N_NODES, recurrent_initializer='orthogonal', return_sequences=True)

        # TimeDistributed fully connected + dropout + output layers to predict at each timestep
        self.fc2 = TimeDistributed(Dense(self.n_fc))
        self.fc2_dropout = TimeDistributed(Dropout(self.DROPOUT))
        self.outputlayer = TimeDistributed(Dense(self.n_output, activation=self.out_activation))

    def call(self, inputs, training=False):
        # inputs expected shape: [batch, height, width, time, channel]
        # Original input shape in example: (256, 9, 11, 100, 1), reshape to batch*time, height, width, channel
        # Rearrange from (B,H,W,T,C) to (B*T,H,W,C) to do conv ops treating time as batch dim
        batch_size, height, width, n_timesteps, channels = inputs.shape
        
        # Defensive cast to int for shapes that may be None in graph mode
        batch_size = tf.shape(inputs)[0] if batch_size is None else batch_size
        height = tf.shape(inputs)[1] if height is None else height
        width = tf.shape(inputs)[2] if width is None else width
        n_timesteps = tf.shape(inputs)[3] if n_timesteps is None else n_timesteps
        channels = tf.shape(inputs)[4] if channels is None else channels

        # Reshape so conv2d can work over spatial dims; TimeDistributed will be simulated manually by flattening time into batch
        x = tf.reshape(inputs, (batch_size * n_timesteps, height, width, channels))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten conv output spatially
        conv_shape = tf.shape(x)
        spatial_dims = conv_shape[1] * conv_shape[2] * conv_shape[3]
        x = tf.reshape(x, (batch_size * n_timesteps, spatial_dims))

        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        # Reshape back to (batch, time, features) for LSTM input
        x = tf.reshape(x, (batch_size, n_timesteps, self.n_fc))

        x = self.lstm1(x)
        x = self.lstm2(x)

        x = self.fc2(x)
        x = self.fc2_dropout(x, training=training)
        output = self.outputlayer(x)

        return output


def my_model_function():
    # Instantiate model with default parameters from fixed example
    return MyModel()


def GetInput():
    # Produces input tensor matching shape expected by MyModel:
    # (batch=256, height=9, width=11, time=100, channel=1)
    return tf.random.uniform((256, 9, 11, 100, 1), dtype=tf.float32)

