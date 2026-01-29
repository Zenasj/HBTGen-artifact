# tf.random.uniform((None, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, **kwargs):
        super(MyModelCheckpoint, self).__init__(**kwargs)

    def on_train_batch_end(self, batch, logs=None):
        # Custom logic to enable save_freq by batch count, injecting 'batch' key safely for formatting
        if (batch + 1) % self.save_freq == 0:
            if logs is None:
                logs = {}
            logs['batch'] = batch + 1
            self._save_model(epoch=self._current_epoch, logs=logs)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reconstructing the original Sequential model as submodules
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='conv_1', input_shape=(32, 32, 3))
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2')
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), name='pool_1')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_3')
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), name='pool_2')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense_1 = tf.keras.layers.Dense(units=32, activation='relu', name='dense_1')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax', name='dense_2')

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.conv_2(x)
        x = self.pool_1(x)
        x = self.bn_2(x, training=training)
        x = self.conv_3(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        return x


def my_model_function():
    model = MyModel()
    # Compile the model with same configuration as original Sequential example
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random float32 tensor with shape (batch_size, H=32, W=32, C=3)
    # Using batch_size=10 as in the original train example
    return tf.random.uniform((10, 32, 32, 3), dtype=tf.float32)

