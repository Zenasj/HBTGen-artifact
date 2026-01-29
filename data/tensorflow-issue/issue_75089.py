# tf.random.uniform((1, 128, 32, 1), dtype=tf.float32) ← Inferred input shape from issue: batch=1, height=128, width=32, channels=1
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture exactly as described in the issue
        # Input shape: (128, 32, 1)
        self.conv1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='valid', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='valid', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='valid', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='valid', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='valid', activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPool2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(0.7)
        self.dense4 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense5 = tf.keras.layers.Dense(10, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.dropout1(x, training=training)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.dropout3(x, training=training)
        x = self.dense3(x)
        x = self.dropout4(x, training=training)
        x = self.dense4(x)
        x = self.dense5(x)
        outputs = self.output_layer(x)
        return outputs


def my_model_function():
    """
    Returns an instance of MyModel.
    Mirrors the architecture and config from the issue's get_model function.
    """
    model = MyModel()
    # Compile with parameters from issue
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model


def GetInput():
    """
    Returns a random input tensor that matches the expected shape for the model.
    Assumption: batch size 1, float32 values.
    """
    return tf.random.uniform(shape=(1, 128, 32, 1), dtype=tf.float32)

