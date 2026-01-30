import math
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

layers = tf.keras.layers


class LayerTest(layers.Layer):
    def __init__(self):
        super(LayerTest, self).__init__()

    def call(self, inputs) -> tf.Tensor:
        predictions = inputs
        
        for k in predictions.keys():
            predictions[k] = tf.math.l2_normalize(predictions[k], axis=-1)

        for step_name in predictions.keys():
            loss = tf.reduce_mean(predictions[k])
            
        return loss
    
    
def Model(target_dim: int = 64):
   
    input_tensor = layers.Input(
        shape=[target_dim], name="input_tensor"
    )
 
    predictions = {'step_0': layers.Lambda(lambda _x: _x)(input_tensor), 'step_1': layers.Lambda(lambda _x: _x)(input_tensor)}
    
    logits = LayerTest()(predictions)

    return keras.Model(inputs=input_tensor, outputs=logits)

               
model = Model()
        
model.compile()
                       
model.summary()

class LayerTest(layers.Layer):
    def __init__(self):
        super(LayerTest, self).__init__()

    def call(self, inputs) -> tf.Tensor:
        predictions = inputs

        predictions2 = {}
        
        for k in predictions.keys():
            predictions2[k] = tf.math.l2_normalize(predictions[k], axis=-1)

        for step_name in predictions2.keys():
            loss = tf.reduce_mean(predictions2[k])
            
        return loss