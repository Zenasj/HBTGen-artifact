# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32) ← Inferred input shape from dataset resize pipeline (images resized to 128x128 RGB)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=5):
        super().__init__()
        # Build the CNN model as per the code in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

def my_model_function():
    # We assume 5 classes (tf_flowers dataset has 5 labels)
    model = MyModel(num_classes=5)

    # Compile the model similar to the setup in the issue
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def GetInput():
    # Return a batch size of 32 with shape (32, 128, 128, 3) and float32 dtype in [0,1] range
    batch_size = 32
    input_tensor = tf.random.uniform((batch_size, 128, 128, 3), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

