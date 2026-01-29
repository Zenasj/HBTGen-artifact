# tf.random.uniform((B=100, H=10), dtype=tf.float32) ← inferred input shape from dataset used in issue

import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Two Dense layers as per original snippet
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

class NestedModel(tf.keras.Model):
    def __init__(self):
        super(NestedModel, self).__init__()
        # Nested BaseModel inside plus an additional Dense layer
        self.base_model = BaseModel()
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.dense(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Encapsulate both models as submodules to reflect issue's focus on nested subclassed models
        self.base_model = BaseModel()
        self.nested_model = NestedModel()

    def call(self, inputs):
        # Forward inputs through both models
        out_base = self.base_model(inputs)
        out_nested = self.nested_model(inputs)
        # Compare outputs elementwise (within a tolerance)
        diff = tf.abs(out_base - out_nested)
        # For demonstration, return boolean tensor where difference < 1e-5
        return diff < 1e-5

def my_model_function():
    # Instantiate MyModel (no pretrained weights)
    return MyModel()

def GetInput():
    # Return random tensor matching expected input shape: [batch_size=2, features=10]
    # batch size = 2 chosen because in issue's dataset batching with batch 2 was used
    return tf.random.uniform(shape=(2, 10), dtype=tf.float32)

