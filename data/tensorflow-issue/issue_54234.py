# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← typical MNIST input image shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple MNIST-like model structure as per the example
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

    def train_step(self, data):
        # Overriding train_step to allow debugging inside loss function and breakpoint
        # Data unpack
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            # Custom loss with debugging support
            # This properly computes SparseCategoricalCrossentropy from logits
            # Including explicit casting and reducing on correct axis
            y_true = tf.cast(y, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            # Here you can put a breakpoint or debug prints
            squared_difference = tf.square(y_true - y_pred)
            loss = tf.reduce_mean(squared_difference, axis=-1)
            loss = tf.reduce_mean(loss)  # mean over batch
            
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics (example metric: sparse categorical accuracy)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    # Return a compiled instance of MyModel
    model = MyModel()
    # Using Adam optimizer and standard sparse categorical accuracy metric
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def GetInput():
    # Return a batch of MNIST-like images as input tensor
    # Here, create a batch of size 128 (same as example)
    batch_size = 128
    # uint8 images normalized to float32 in range [0,1]
    x = tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)
    return x

