# tf.random.uniform((BATCH_SIZE, input_shape)) where BATCH_SIZE=32 (inferred from code), input_shape = embeddings.shape[1]

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Because the issue was around classic dense Softmax classifier with dropout,
        # we reproduce the same structure as given in SoftMax.build()
        self.dense1 = layers.Dense(1024, activation='relu', input_shape=(input_shape,))
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(1024, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

        # We build the model to create weights properly
        self.build((None, input_shape))

        # Setup loss and optimizer for potential training (mimicking original build)
        self.loss_fn = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        
    def call(self, inputs, training=False):
        # Inputs assumed dense. If sparse input accidentally passed, convert to dense with .to_dense()
        # But since doc and fixes recommend converting inputs to dense beforehand, we expect dense inputs.
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

def my_model_function():
    # For demonstration we pick input_shape=512 as typical face embedding size,
    # and num_classes=10 as placeholder (since labels count unknown here).
    # These can be adjusted based on actual embeddings.
    input_shape = 512
    num_classes = 10
    model = MyModel(input_shape=input_shape, num_classes=num_classes)
    return model

def GetInput():
    # Create a random input tensor shaped like embeddings input,
    # batch size 32 (common batch size from the original code), embedding dim 512 (typical)
    BATCH_SIZE = 32
    EMBEDDING_DIM = 512
    # Use float32 to match typical model input type
    return tf.random.uniform((BATCH_SIZE, EMBEDDING_DIM), dtype=tf.float32)

