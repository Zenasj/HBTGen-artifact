# tf.random.uniform((B, 100), dtype=tf.float32) ← inferred input shape from MLP example: (batch, 100 features)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Re-implement a small sequential MLP similar to testing_utils.get_small_sequential_mlp
        # Assumption: 2 hidden layers, 10 output classes (matching the example)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

class TestCallback(tf.keras.callbacks.Callback):
    def set_model(self, model):
        """Check whether optimizer operations are present in the graph.

        This mimics the callback behavior that verifies if the model graph
        is fully populated when set_model is called, focusing on the optimizer ops.
        """
        # Compose name scope where optimizer ops should reside
        optimizer_name_scope = 'training/' + model.optimizer.__class__.__name__
        try:
            # TF2 with eager enabled doesn't have default graph like TF1, but for 
            # compatibility with graphs in graph mode, get default graph (if exists).
            graph = tf.compat.v1.get_default_graph()
            graph_def = graph.as_graph_def()
        except Exception:
            # In eager mode, can't get graph_def -> we treat as graph built
            return
        
        # Search nodes for optimizer ops namespace prefix
        for node in graph_def.node:
            if node.name.startswith(optimizer_name_scope):
                return
        # If not found, raise runtime error
        raise RuntimeError(
            'The optimizer operations are not present in the model graph '
            'when the Callback.set_model function is called.'
        )

def my_model_function():
    model = MyModel()
    # Compile to add optimizer and loss matching the examples
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape:
    # (batch_size=10, features=100)
    # Use tf.random.uniform for generality (like the comment)
    x = tf.random.uniform((10, 100), dtype=tf.float32)
    return x

