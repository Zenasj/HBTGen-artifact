from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf                                                        
                                                                               
class TestModel(tf.keras.models.Model):
  @tf.function
  def test(self):
    return 123

test_model = TestModel()
signatures = [test_model.test.get_concrete_function()]

converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

import tensorflow as tf                                                        
                                                                               
class TestModel(tf.keras.models.Model):                                        
  @tf.function                                                                 
  def test_1(self):                                                            
    return 123                                                                 
                                                                               
  @tf.function                                                                 
  def test_2(self):                                                            
    return 456                                                                 
                                                                               
test_model = TestModel()                                                       
signatures = [                                                                 
  test_model.test_1.get_concrete_function(),                                   
  test_model.test_2.get_concrete_function(),                                   
]                                                                              
                                                                               
converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]                           

# This now works but...
tflite_model = converter.convert()

# This throws "ValueError: NULL SignatureDef inputs for exported method test_1"
interpreter = tf.lite.Interpreter(model_content=tflite_model)

import tensorflow as tf                                                        
                                                                               
class TestModel(tf.keras.models.Model):
  @tf.function
  def test_1(self, dummy):                                                     
    return 123                                                                 
                                                                               
  @tf.function                                                                 
  def test_2(self, dummy):
    return 456                                                                 
  
test_model = TestModel()
signatures = [
  test_model.test_1.get_concrete_function(dummy=tf.constant(0)),               
  test_model.test_2.get_concrete_function(dummy=tf.constant(0)),               
] 
    
converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]                           
tflite_model = converter.convert()
    
interpreter = tf.lite.Interpreter(model_content=tflite_model)

# This works, but requires explicitly setting the dummy, which I'm trying to avoid.
interpreter.get_signature_runner('test_1')(dummy=tf.constant(0))

# Throws "ValueError: Invalid number of inputs provided for running a SignatureDef, expected 1 vs provided 0".
interpreter.get_signature_runner('test_1')()