# tf.random.uniform((B, 100), dtype=tf.float32) ← Inferred input shape is (batch_size, 100) based on usage in example

import tensorflow as tf

class DenseNormLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation, name=None):
        super(DenseNormLayer, self).__init__(name=name)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense_layer = tf.keras.layers.Dense(units=units, activation=activation)
    
    def call(self, x, training=None):
        x = self.batch_norm(x, training=training)
        x = self.dense_layer(x)
        return x

class BaselineModel(tf.keras.Model):
    def __init__(self, targets, name="BaselineModel"):
        super(BaselineModel, self).__init__(name=name)
        # Layers initialized in call-graph order for clarity, but summary order will follow creation order
        self.block1 = DenseNormLayer(units=1024, activation="relu", name="block1")
        self.block2 = DenseNormLayer(units=1024, activation="relu", name="block2")
        self.d1 = tf.keras.layers.Dense(units=512, activation="relu", name="d1")
        self.dp = tf.keras.layers.Dropout(rate=0.2, name="dropout_layer")  # Correct order per call()
        self.d2 = tf.keras.layers.Dense(units=1024, activation="relu", name="d2")
        self.sigmoid_layer = tf.keras.layers.Dense(units=targets, activation="sigmoid", name="sigmoid_layer")
    
    def call(self, X, training=None):
        x = self.block1(X, training=training)
        x = self.block2(x, training=training)
        x = self.d1(x)
        x = self.dp(x, training=training)
        x = self.d2(x)
        x = self.sigmoid_layer(x)
        return x

    def build_graph(self, dim):
        x = tf.keras.Input(shape=dim)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

# Single model class as required
class MyModel(tf.keras.Model):
    def __init__(self, targets=100):
        super(MyModel, self).__init__(name="MyModel")
        # Embed the BaselineModel as a submodule
        self.baseline = BaselineModel(targets=targets)
    
    def call(self, inputs, training=None):
        # Simply forward inputs through baseline model
        return self.baseline(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with default 100 output targets
    return MyModel()

def GetInput():
    # Return a random float32 tensor that matches the input expected by MyModel: (batch_size, 100)
    # Using batch size = 4 as an example
    return tf.random.uniform((4, 100), dtype=tf.float32)

