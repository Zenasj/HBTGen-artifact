# tf.random.uniform((B, 14, 1), dtype=tf.float32) ← Input shape inferred from dataset X_train shape (samples, 14, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # GRU stack modeled on buildManyToOneModel function
        self.gru1 = tf.keras.layers.GRU(32, 
                                        return_sequences=True, 
                                        input_shape=(14, 1))
        self.gru2 = tf.keras.layers.GRU(64, return_sequences=True)
        self.gru3 = tf.keras.layers.GRU(128, return_sequences=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.gru3(x)
        x = self.batch_norm(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Instantiate compiled model with same compile config as buildManyToOneModel
    model = MyModel()
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    return model

def GetInput():
    # Generate a random float32 tensor with shape (batch_size=32, 14 time steps, 1 feature)
    # This matches the model's expected input shape, based on X_train shape in the original code
    B = 32  # arbitrary batch size for input generation
    H = 14  # time steps dimension from data shape
    W = 1   # feature dimension from data shape (after adding new axis)
    input_tensor = tf.random.uniform((B, H, W), dtype=tf.float32)
    return input_tensor

