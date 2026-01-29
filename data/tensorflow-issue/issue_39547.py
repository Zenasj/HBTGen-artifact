# tf.random.normal((1, 5)) ← Input tensor shape is (batch_size=1, feature_dim=5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        # The issue discussed involves the 'training' argument and dropout behavior.
        # Show messages to indicate training mode based on the training argument exactly as in the issue.
        if training is True:
            # Dropout should only apply during training mode
            tf.print("in training")
            x = self.dropout(x, training=training)
        elif training is None:
            tf.print("training None")
        else:
            tf.print("not in training")
        return self.dense2(x)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random normal tensor shaped (1,5), matching the example with batched input in the issue
    return tf.random.normal((1, 5))

