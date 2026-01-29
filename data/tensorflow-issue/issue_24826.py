# tf.random.uniform((B, ...), dtype=tf.float32)  ← Input shape is ambiguous in the issue; assuming batch dimension only.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The issue discusses estimator._call_model_fn which returns an EstimatorSpec including model graph
        # In Keras style, we mimic a simple model representing the "model_fn" build step that typically
        # is expected to build the graph once per training call.

        # Since the original code discusses _call_model_fn called every train step vs once,
        # we model an initialization phase and a "train step" phase separately here.

        # This dummy sub-model symbolizes "graph construction" / model body.
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # Example output size of 10 classes

        # Flag to simulate "initialized once" graph
        self._init_done = False

    def call(self, inputs, training=False):
        # If training=True and initialization not done => initialize the graph
        if training and not self._init_done:
            self._initialize_graph()
            self._init_done = True

        # Model forward pass
        x = self.dense1(inputs)
        logits = self.dense2(x)
        return logits

    def _initialize_graph(self):
        # Dummy initialization logic representing graph build once at start of training
        # In real Estimator, this would be the place where _call_model_fn is executed once to build graph.
        # Here, it is just a placeholder to illustrate the difference.
        pass

def my_model_function():
    # Instantiate MyModel. 
    # In an estimator scenario, this function would correspond to "building the graph once".
    return MyModel()

def GetInput():
    # Since the input shape isn't specified in the issue, assume a batch of feature vectors with size 32
    # For example, input shape (batch_size=4, feature_dim=20)
    batch_size = 4
    feature_dim = 20
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

