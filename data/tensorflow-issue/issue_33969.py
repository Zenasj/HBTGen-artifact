import tensorflow as tf
from tensorflow.keras import layers

# Create a simple functional model
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
  def __init__(self, output_dim, name=None):
    super(SubclassedModel, self).__init__(name=name)
    self.output_dim = output_dim
    self.dense_1 = keras.layers.Dense(64, activation='relu', name='dense_1')
    self.dense_2 = keras.layers.Dense(64, activation='relu', name='dense_2')
    self.dense_3 = keras.layers.Dense(output_dim, name='predictions')
  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    x = self.dense_3(x)
    return x
  def get_config(self):
    return {'output_dim': self.output_dim, 'name': self.name}

subclassed_model = SubclassedModel(10)
# Call the subclassed model once to create the weights.
subclassed_model(tf.ones((1, 784)))

# Copy weights from functional_model to subclassed_model.
subclassed_model.set_weights(functional_model.get_weights())