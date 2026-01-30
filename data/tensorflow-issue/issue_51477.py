import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Lookup(tf.keras.layers.Layer):
    def build(self, input_shape):
        names = tf.constant(["a", "b"])
        numbers = tf.constant([1, 2], dtype=tf.int64)
        
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(names, numbers), -1)
        self.built = True
        
    def call(self, names):
        return self.table.lookup(tf.reshape(names, [-1]))

with tf.control_dependencies([tf.compat.v1.tables_initializer()]):  
    names = tf.keras.Input(shape=(2,), dtype=tf.string, name='names')
    model_outputs = Lookup()(names)
    model = tf.keras.Model(
        inputs=[names],
        outputs=model_outputs,
    )
    
model.save('./export')

converter = tf.lite.TFLiteConverter.from_saved_model('./export')
tflite_model = converter.convert()
with open('simple.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='simple.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], np.array([['a', 'b']]))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('output', output_data)

-1
-1