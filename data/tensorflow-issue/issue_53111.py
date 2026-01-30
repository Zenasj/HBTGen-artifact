from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf                                                        
                                                                               
class TestModel(tf.keras.models.Model):                                        
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)                                                 
    self._hash = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(['testing', 'this', 'thing']),                         
            tf.constant([1, 2, 3])),
        default_value=-1)                                                      

  @tf.function
  def test(self, word):
    return self._hash.lookup(word)                                             


test_model = TestModel()                                                       
signatures = [test_model.test.get_concrete_function(tf.TensorSpec([None], tf.string))] 
    
converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]                           
converter.target_spec.supported_ops = [                                        
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS                                                 
]

tflite_model = converter.convert()                                             
interpreter = tf.lite.Interpreter(model_content=tflite_model)

# Causes segmentation fault. Running test_model.test directly works fine.
result = interpreter.get_signature_runner()(word=tf.constant(['testing', 'that', 'thing']))