# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)
import math
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the model architecture from the issue reproducing code:
        # Input shape is (32, 32, 3) - CIFAR-10 sample images normalized to float32

        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3))
        self.act1 = tf.keras.layers.Activation('relu')
        
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        
        self.conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        output = self.dense2(x)
        return output


def my_model_function():
    # Initialize and compile with same optimizer-loss as original
    model = MyModel()
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[])
    return model


def GetInput():
    # Generate a batch of random inputs that match CIFAR-10 image shape:
    # The example used batch_size=32, shape (32, 32, 3), dtype float32 normalized [0,1].
    batch_size = 32
    input_shape = (batch_size, 32, 32, 3)
    # Using uniform random to simulate image inputs normalized to [0,1].
    return tf.random.uniform(input_shape, minval=0.0, maxval=1.0, dtype=tf.float32)

