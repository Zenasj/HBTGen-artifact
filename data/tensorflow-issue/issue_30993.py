# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image shape (batch size B inferred)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture extracted from the example, matching CIFAR-10 input (32x32 RGB images)
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(32,32,3))
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop2 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop3(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Returns an untrained instance of MyModel
    model = MyModel()
    # Compile with RMSProp and categorical crossentropy matching the example
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def GetInput():
    # Returns a random float32 input tensor consistent with CIFAR-10 shape (batch size 32)
    # Normalize to [0, 1] as in the example
    batch_size = 32
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

