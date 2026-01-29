# tf.random.uniform((B, 5), dtype=tf.float32) ← Based on df with 5 features (columns) from the issue description

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimicking the example sequential model from the issue comments:
        # 3 dense layers: two with 10 units ReLU, final with 1 unit sigmoid
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Instantiate and compile the model as per the example in the issue
    model = MyModel()
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'acc'])
    return model

def GetInput():
    # The input is expected to be a tensor of shape (batch_size, 5)
    # as the dataframe had 5 columns/features
    # Generate random float32 inputs to simulate this
    batch_size = 8  # arbitrary batch size
    feature_dim = 5
    # Returns a tensor with shape (batch_size, 5)
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

