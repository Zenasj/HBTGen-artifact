# tf.random.uniform((1, 64, 12, 86, 98, 8), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Dice coefficient metric and loss functions as provided
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.cast(K.flatten(y_pred), dtype='float32')
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) * smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred, smooth=1):
    return -dice_coef(y_true, y_pred)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We assume the model works on inputs of shape (64, 12, 86, 98, 8)
        # This comes from the extracted shape of outputs after processing
        # So input shape to the model will exclude batch dimension:
        self.input_shape_ = (64, 12, 86, 98, 8)
        
        # Build network layers explicitly (Dense layers expect last dim as feature dim)
        # We will flatten input except for last dim, apply Dense(128), then Dense(1)
        
        # To do so, flatten all dims except last, apply Dense layers on last dimension/features
        # or apply Dense to each spatial location independently on channels.
        # Since the original code applied Dense directly on Input of shape with 6+ dims (causing error),
        # We must reshape before Dense layers.
        
        # Here we implement a relearning logic:
        # Flatten spatial dims into batch dimension, then apply Dense over features (last dim).
        self.flatten_dims = 64*12*86*98  # total spatial dims flattened
        self.dense1 = Dense(128, activation='elu')
        self.dense2 = Dense(1, activation='elu')

    def call(self, inputs):
        # inputs shape: (batch_size, 64, 12, 86, 98, 8)
        # flatten dims except last:
        batch_size = tf.shape(inputs)[0]
        # Reshape to (batch_size * flatten_dims, features)
        x = tf.reshape(inputs, (batch_size * self.flatten_dims, self.input_shape_[-1]))  
        x = self.dense1(x)
        x = self.dense2(x)
        # The output shape is (batch_size * flatten_dims, 1)
        # Reshape back to (batch_size, 64, 12, 86, 98, 1)
        x = tf.reshape(x, (batch_size,) + self.input_shape_[:-1] + (1,))
        return x

def my_model_function():
    # Instantiate and compile the model with dice loss and dice coef metric
    model = MyModel()
    # Use Adam optimizer by default
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=dice_loss,
                  optimizer=optimizer,
                  metrics=[dice_coef])
    return model

def GetInput():
    # Return an input tensor matching expected input: shape (batch, 64, 12, 86, 98, 8)
    # Using batch=1 for simplicity as in original code
    # Use tf.random.uniform with dtype=tf.float32
    return tf.random.uniform((1, 64, 12, 86, 98, 8), dtype=tf.float32)

