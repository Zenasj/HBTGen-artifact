import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class RecurrentConfig(BaseLayer):
    '''Basic configurable recurrent layer'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers: List[layers.Layer] = stack_layers(self.params,
                                                       self.num_layers,
                                                       self.layer_name)

    def call(self, inputs: np.ndarray) -> layers.Layer:
        '''This function is a sequential/functional call to this layers logic
        Args:
            inputs: Array to be processed within this layer
        Returns:
            inputs processed through this layer'''
        processed = inputs
        for layer in self.layers:
            processed = layer(processed)
        return processed

    @staticmethod
    def default_params() -> Dict[Any, Any]:
        return{
            'units': 32,
            'recurrent_initializer': 'glorot_uniform',
            'dropout': 0,
            'recurrent_dropout': 0,
            'activation': None,
            'return_sequences': True
        }

import tensorflow as tf
class DummyLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = tf.keras.layers.LSTM(2)
    
    def call(inputs):
        return self.a(inputs)
    
tf.keras.layers.Bidirectional(DummyLayer())

import tensorflow as tf
class DummyLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(go_backwards=True, *args, **kwargs)
        #self.a = tf.keras.layers.LSTM(2, go_backwards=True)
    
    def call(inputs):
        return None#self.a(inputs)
    
tf.keras.layers.Bidirectional(DummyLayer())

import tensorflow as tf

class DummyLayer(tf.keras.layers.Layer):
  def __init__(self, go_backwards=False, *args, **kwargs):
    super(DummyLayer, self).__init__(*args, **kwargs)
    self.go_backwards = go_backwards
    self.a = tf.keras.layers.LSTM(2, go_backwards=self.go_backwards)

  def call(self, inputs):
    return self.a(inputs)

  @property
  def return_sequences(self):
    return self.a.return_sequences

  @property
  def return_state(self):
    return self.a.return_state

  def get_config(self):
    config = {'go_backwards': self.go_backwards}
    base_config = super(DummyLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

tf.keras.layers.Bidirectional(DummyLayer())