# tf.random.uniform((42, 100, 10), dtype=tf.float32)  ← Assumed input shape: batch_size=42, window_size=100, features=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, window_size=100, features=10, classes=5):
        super().__init__()
        # Batch normalization to normalize input data
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        # Two convolutional layers with ReLU and batch normalization
        self.conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=3)
        self.act1 = tf.keras.layers.Activation('relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=6)
        self.act2 = tf.keras.layers.Activation('relu')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

        # Three LSTM layers with L1L2 recurrent regularizers as in the original example
        self.lstm1 = tf.keras.layers.LSTM(
            32,
            return_sequences=True,
            recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))
        self.lstm2 = tf.keras.layers.LSTM(
            24,
            return_sequences=True,
            recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))
        self.lstm3 = tf.keras.layers.LSTM(
            16,
            return_sequences=False,
            recurrent_regularizer=tf.keras.regularizers.L1L2(0.5, 0.1))

        # Dense layer with ReLU and Dropout
        self.dense1 = tf.keras.layers.Dense(8)
        self.act3 = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.35)

        # Output layer with softmax activation for classification
        self.output_layer = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.batch_norm1(inputs, training=training)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.batch_norm2(x, training=training)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.batch_norm3(x, training=training)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense1(x)
        x = self.act3(x)
        x = self.dropout(x, training=training)
        out = self.output_layer(x)
        return out


def my_model_function():
    # Instantiate the model with some default config matching the example
    # Assumptions about input shape and number of classes must be consistent with GetInput()
    window_size = 100  # assumed sequence length
    features = 10      # assumed input feature dimension
    classes = 5        # assumed number of output classes
    model = MyModel(window_size=window_size, features=features, classes=classes)

    # Compile model with Adam optimizer and categorical crossentropy loss as per original example
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def GetInput():
    # Generate a random tensor matching the expected model input shape
    # batch_size=42 as per original input layer specification
    batch_size = 42
    window_size = 100
    features = 10
    # dtype float32 assumed, consistent with typical Keras input dtype
    return tf.random.uniform((batch_size, window_size, features), dtype=tf.float32)

