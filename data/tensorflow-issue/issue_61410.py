from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

ly = tf.keras.layers
input8 = tf.keras.Input(shape=(8, 8, 1))

y = ly.Conv2D(filters=1, kernel_size=[3, 3], kernel_initializer=tf.keras.initializers.Constant(1), name="conv_1")(input8)
y = ly.Conv2D(filters=1, kernel_size=[5, 5], kernel_initializer=tf.keras.initializers.Constant(1), name="conv_2")(y)
upsample_output = ly.UpSampling2D(size=(2,2))(y)
upsample_model = tf.keras.Model(inputs=input8, outputs=upsample_output)

x = ly.Conv2D(filters=1, kernel_size=[3, 3], kernel_initializer=tf.keras.initializers.Constant(1), name="conv_1")(input8)
x = ly.Conv2D(filters=1, kernel_size=[5, 5], kernel_initializer=tf.keras.initializers.Constant(1), name="conv_2")(x)
transpose_output = ly.Conv2DTranspose(filters=1, kernel_size=[2, 2], kernel_initializer=tf.keras.initializers.Constant(1), strides=[2,2], name="conv_transpose_1")(x)
transpose_model = tf.keras.Model(inputs=input8, outputs=transpose_output)

converter = tf.lite.TFLiteConverter.from_keras_model(upsample_model)
upsample_tflite_model = converter.convert()
with open('upsample_model.tflite', 'wb') as handle:
    handle.write(upsample_tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(transpose_model)
transpose_tflite_model = converter.convert()
with open('transpose_model.tflite', 'wb') as handle:
    handle.write(transpose_tflite_model)

upsample_interpreter = tf.lite.Interpreter(model_path='models/upsample_model.tflite')
transpose_interpreter = tf.lite.Interpreter(model_path='models/transpose_model.tflite')

print("BEFORE INPUT RESIZING (expect input shape (1, 8, 8, 1) and output shape (1, 4, 4, 1)")
print()
print(upsample_interpreter.get_input_details())
print(upsample_interpreter.get_output_details())
print()
print()
print(transpose_interpreter.get_input_details())
print(transpose_interpreter.get_output_details())
print()
print()

upsample_interpreter.resize_tensor_input(0, (1, 16, 16, 1))
transpose_interpreter.resize_tensor_input(0, (1, 16, 16, 1))
upsample_interpreter.allocate_tensors()
transpose_interpreter.allocate_tensors()

print("AFTER INPUT RESIZING (expect input shape (1, 16, 16, 1) and output shape (1, 8, 8, 1)")
print()
print(upsample_interpreter.get_input_details())
print(upsample_interpreter.get_output_details())
print()
print()
print(transpose_interpreter.get_input_details())
print(transpose_interpreter.get_output_details())