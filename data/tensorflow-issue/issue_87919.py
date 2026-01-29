# tf.random.uniform((B, 128), dtype=tf.float32) ← The exact input shape is dynamic in original code,
# but after one-hot encoding categorical features + numeric scaled features,
# the input vector length is fixed per sample.
# Since exact number of categorical columns and numeric columns are unknown,
# we approximate input shape as (batch_size, input_dim) where input_dim is fixed.
# For demonstration, we'll assume input_dim = 100 as a placeholder.
# This should be adjusted to the actual sum of one-hot categories and numeric features.

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class MyModel(tf.keras.Model):
    def __init__(self, input_dim=100, num_classes=3):
        super().__init__()
        # Input layer does not need to be explicitly defined here,
        # as we will rely on the input tensor shape at call time.
        # Dense layers defined as per issue description:
        self.dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.3)
        
        self.dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)
        
        self.out_layer = Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        output = self.out_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default input_dim placeholder
    return MyModel(input_dim=100, num_classes=3)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Since input shape is (batch_size, input_dim), batch_size is chosen arbitrarily here.
    batch_size = 16  # arbitrary batch size to test forward pass
    input_dim = 100  # must match model's input_dim parameter
    # Random uniform input simulating combined one-hot plus scaled numeric features
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

