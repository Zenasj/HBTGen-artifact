# tf.random.uniform((batch_size, 15), dtype=tf.float32) ← Input shape inferred as (batch_size, 15) based on parsed CSV features

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model configuration inferred from issue:
        # - Input dimension: 15 features
        # - Dense layers with 'selu' activation
        # - 5 layers of 40 units each (based on example hyperparameters)
        # - Output layer with 9 units (for 9 classes) and softmax activation
        
        self.layerdensity = 40
        self.amount_of_layers = 5
        self.output_units = 9
        
        # First Dense layer with input_dim=15
        self.dense_layers = []
        self.dense_layers.append(Dense(self.layerdensity, activation=tf.nn.selu, input_shape=(15,)))
        
        # Remaining dense layers (amount_of_layers - 1)
        for _ in range(self.amount_of_layers - 1):
            self.dense_layers.append(Dense(self.layerdensity, activation=tf.nn.selu))
        
        # Output layer
        self.output_layer = Dense(self.output_units, activation=tf.nn.softmax, name="Output")
        
        # Build Sequential model internally for convenience
        self.model = Sequential(self.dense_layers + [self.output_layer])
    
    def call(self, inputs, training=False):
        # Forward pass through the Sequential model
        return self.model(inputs)

def my_model_function():
    # Return an instance of MyModel with all weights initialized randomly.
    # No pretrained weights available in original code snippet.
    return MyModel()

def GetInput():
    # Return a random input tensor simulating the parsed input features.
    # Input shape: (batch_size, 15)
    # Batch size inferred from typical badge sizes used (e.g. 32 in given dataset)
    batch_size = 32
    
    # Generate uniform random floats in [0,1) as dummy input features
    # dtype float32 to match Keras default
    return tf.random.uniform((batch_size, 15), dtype=tf.float32)

