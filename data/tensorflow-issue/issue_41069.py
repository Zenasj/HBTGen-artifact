# tf.random.uniform((1, 200, 200, 3), dtype=tf.float32) <- Inferred input shape and dtype based on MobileNet input_shape=(200,200,3)

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load MobileNet base model without top layers, with pretrained weights
        # input_shape must include batch dimension at runtime; here (200,200,3)
        self.base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(200, 200, 3), dropout=0.2)
        
        # First output branch layers
        self.pool1 = GlobalAveragePooling2D()
        self.dense1_1 = Dense(1024, activation='relu')
        self.dense1_2 = Dense(1024, activation='relu')
        self.dense1_3 = Dense(512, activation='relu')
        self.out_types = Dense(20, activation='softmax', name='Output1')
        
        # Second output branch layers
        self.pool2 = GlobalAveragePooling2D()
        self.dense2_1 = Dense(512, activation='relu')
        self.dense2_2 = Dense(1024, activation='relu')
        self.dense2_3 = Dense(512, activation='relu')
        self.out_values = Dense(3, activation='sigmoid', name='Output2')

    def call(self, inputs, training=False):
        # Forward pass through base model
        x = self.base_model(inputs, training=training)
        
        # First output branch
        y1 = self.pool1(x)
        y1 = self.dense1_1(y1)
        y1 = self.dense1_2(y1)
        y1 = self.dense1_3(y1)
        out1 = self.out_types(y1)
        
        # Second output branch
        y2 = self.pool2(x)
        y2 = self.dense2_1(y2)
        y2 = self.dense2_2(y2)
        y2 = self.dense2_3(y2)
        out2 = self.out_values(y2)
        
        # Return tuple of outputs matching original multi-output model
        return out1, out2

def my_model_function():
    # Return an instance of MyModel with pretrained MobileNet base weights
    # User can load weights later if needed
    model = MyModel()
    return model

def GetInput():
    # Return random input tensor matching (batch_size=1, 200, 200, 3) and dtype float32
    # Matches the MobileNet expected input size from the issue
    return tf.random.uniform((1, 200, 200, 3), dtype=tf.float32)

