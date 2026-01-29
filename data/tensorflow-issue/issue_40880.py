# tf.random.uniform((B, 4), dtype=tf.float32) ← Input shape inferred from model: batch size B, features=4

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # Note: Using softmax with 1 output unit is unusual; replicate original logic.
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.softmax)

    def compile(self, optimizer, loss, metric):
        """
        Overriden compile to initialize optimizer, loss, and metric for custom train_step.
        Using run_eagerly=True here to ensure eager execution within train_step as discussed.
        """
        super().compile(run_eagerly=True)
        self.opt = optimizer
        self.loss = loss
        self.metric = metric  
    
    def train_step(self, data):
        # Confirm eager execution mode inside train_step (expected True due to run_eagerly=True above)
        print(f"Eager execution mode: {tf.executing_eagerly()}")
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = self.loss(y, y_pred)
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        metric = self.metric(y, y_pred)
        return {"loss": loss, "metric": metric}
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate a random tensor matching input shape [batch_size, 4]
    # Batch size chosen as 10 to match example usage
    return tf.random.uniform((10, 4), dtype=tf.float32)

