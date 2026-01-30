from tensorflow import keras
from tensorflow.keras import layers

import copy
import tensorflow as tf

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=1)
    def call(self, input):
        """Run the model."""
        result = self.dense1(input)
        result = self.dense2(result)
        return result
    def get_config(self):
        config = []
        for layer in self.layers:
            config.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        return copy.deepcopy(config)

class NestedModel(tf.keras.Model):
    def __init__(self):
        super(NestedModel, self).__init__()
        self.base_model = BaseModel()
        self.dense = tf.keras.layers.Dense(units=1)
    def call(self, input):
        result = self.base_model(input)
        result = self.dense(result)
        return result
    def get_config(self):
        config = []
        for layer in self.layers:
            config.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        return copy.deepcopy(config)

model = NestedModel()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), loss='mean_squared_error', metrics=['accuracy'])
estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
estimator.train(input_fn=lambda: tf.data.Dataset.from_tensor_slices((tf.random_uniform([100, 10]), tf.random_uniform([100, ]))).batch(2).repeat(10))