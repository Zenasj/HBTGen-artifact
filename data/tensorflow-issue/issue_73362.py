# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape (batch_size, 10) as per example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the Sequential architecture from the issue reproduction code
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(10,))
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.dense4(x)
        x = self.dropout4(x, training=training)
        return self.output_layer(x)

def my_model_function():
    model = MyModel()
    # Compile as per original example
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor shaped like (batch_size, 10), batch_size = 32 as a common default
    return tf.random.uniform((32, 10), dtype=tf.float32)

