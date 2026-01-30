from tensorflow.keras import layers

import tensorflow as tf
import keras

input0_shape = [2, 2, 1]
input1_shape = [1, 2, 2, 1]
output_shape = [1, 2, 2, 1]

tf_input0 = keras.Input(input0_shape[1:], batch_size=input0_shape[0])
tf_input1 = keras.Input(input1_shape[1:], batch_size=input1_shape[0])


class MyMatMul(keras.layers.Layer):
    def call(self, tf_input0, tf_input1):
        tf_output = tf_input0 * tf_input1
        return tf_output

tf_output = MyMatMul()(tf_input0, tf_input1)

model = keras.Model(inputs=[tf_input0, tf_input1], outputs=[tf_output])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

# OK                                                  
# (5, 352, 352, 3), (352, 352, 3) -> (5, 352, 352, 3) 
# (2, 1, 1, 3), (1, 3) -> (2, 1, 1, 3)                
# (2, 1, 1, 3), (2, 3) -> (2, 1, 2, 3)                
# (2, 1, 1, 3), (3, 3) -> (2, 1, 3, 3)                
# (2, 2, 2, 1), (2, 1) -> (2, 2, 2, 1)                
# (2, 2, 1), (2, 1) -> (2, 2, 1)                      
# (3, 2), (2,) -> (3, 2)                              
# (3,) -> (1, 2, 2, 3) -> (1, 2, 2, 3)                
# (2,), (1, 2) -> (1, 2)                              
# (2, 2), (1, 2) -> (2, 2)                            
                                                      
# NG                                                  
# (1, 2, 2, 3), (3,) -> (1, 2, 2, 3)                  
# (1, 2, 2, 1), (2, 2, 1) -> (1, 2, 2, 1)             
# (2, 2, 1), (1, 2, 2, 1) -> (1, 2, 2, 1)             
# (2, 1), (1, 2, 1) -> (1, 2, 1)                      
# (1, 2, 2, 1), (2, 1) -> (1, 2, 2, 1)                
# (1, 2, 1), (2, 1) -> (1, 2, 1)                      
# (1, 2), (2,) -> (1, 2)