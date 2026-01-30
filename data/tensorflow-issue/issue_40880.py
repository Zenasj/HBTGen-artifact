import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

Python
import tensorflow as tf
import numpy as np
print(tf.__version__) # Has to be 2.2

# Example modified from: https://keras.io/api/models/model/
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.softmax)

    def compile(self, optimizer, loss, metric):
        """
        Overriden .compile() method to initialize
        optimizer, loss, and metric for train_step function
        """
        super().compile()
        self.opt = optimizer
        self.loss = loss
        self.metric = metric  
    
    def train_step(self, data):
        print(f"Eager execution mode: {tf.executing_eagerly()}")
        X, y = data
        
        # Track gradients
        with tf.GradientTape() as tape:
             y_pred = self.call(X)
             loss = self.loss(y, y_pred)
        
        # Compute and modify the weights
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))        

        # Compute metric
        metric = self.metric(y, y_pred)
        return {"loss": loss, "metric": metric}
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

if __name__ == "__main__":
    # Should be True
    print(f"Eager execution mode: {tf.executing_eagerly()}")

    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, (10, 1))
    model = MyModel()
    model.compile(tf.keras.optimizers.Adam(), 
                tf.keras.losses.BinaryCrossentropy(),
                tf.keras.metrics.Accuracy())
    # Should print True
    model.fit(X, y)