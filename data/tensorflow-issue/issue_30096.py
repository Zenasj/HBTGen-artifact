# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset preprocessing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Replicating the model architecture from the issue
        # Note: BatchNormalization axis=3 for channels last, matching input shape (28,28,1)
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='initial_padding')
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=8,
            padding='same',
            name='conv_layer'
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=3, name='bn_layer')
        self.activation = tf.keras.layers.Activation('relu', name='activation_layer')
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten(name='flatten_layer')
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense_layer')
        self.dropout = tf.keras.layers.Dropout(0.2, name='dropout_layer')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.zero_padding(inputs)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly as in the issue
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor input matching model input shape (batch_size, 28, 28, 1)
    # We use batch_size = 32 as a reasonable default
    return tf.random.uniform(shape=(32, 28, 28, 1), dtype=tf.float32)

