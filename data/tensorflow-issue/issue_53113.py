import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf                                                        
                                                                               
class TestModel(tf.keras.models.Model):                                        
                                                                               
  def __init__(self, dims, **kwargs):                                          
    super().__init__(**kwargs)                                                 
    self._dense = tf.keras.layers.Dense(dims)                                  
        
  @tf.function
  def test(self, x):
    return self._dense(x)                                                      
                                                                               
  
# Fails only if dims=0.
test_model = TestModel(dims=0)                                                 
signatures = [
  test_model.test.get_concrete_function(x=tf.TensorSpec([None, 10], tf.float32))  
]                                                                              
    
converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model)                                                                            
converter.optimizations = [tf.lite.Optimize.DEFAULT]                           
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]         

# Fails with "ConverterError: Quantize weights transformation failed."         
# But calling directly test_model.test(tf.random.normal(shape=[1, 10]) works.
tflite_model = converter.convert()
  
interpreter = tf.lite.Interpreter(model_content=tflite_model)                  
result = interpreter.get_signature_runner()(x=tf.random.normal(shape=[1, 10]))