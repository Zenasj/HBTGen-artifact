from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def quant(x, num_bits=8):
    return tf.quantization.fake_quant_with_min_max_args(x, -1, 1, num_bits, False)

class QConv(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, weight_quantizer, activation_quantizer):
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        super().__init__(filters = filters, kernel_size = kernel_size)

    def call(self, bottom):
        return self.activation_quantizer(self.convolution_op(bottom, self.weight_quantizer(self.kernel)))

tf.keras.backend.set_image_data_format("channels_last")
input_tensor = tf.keras.Input(shape=(64, 64, 3), batch_size=1)
quantized_input_tensor = quant(input_tensor, num_bits=8)

# 8->8bit convolution
layer = QConv(filters=32, kernel_size=3, weight_quantizer=quant, activation_quantizer=lambda x: quant(x, num_bits=8))
output8 = layer(quantized_input_tensor)

# 8->16bit convolution
layer = QConv(filters=32, kernel_size=3, weight_quantizer=quant, activation_quantizer=lambda x: quant(x, num_bits=16))
output16 = layer(quantized_input_tensor)

model = tf.keras.Model(inputs=[input_tensor], outputs=[output8, output16])

train_save_path = "/tmp/debug_model"
convert_model_path = "/tmp/converted.tflite"

model.save(train_save_path)
converter = tf.lite.TFLiteConverter.from_saved_model(train_save_path)
converter.optimizations =[tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int16
tflite_model = converter.convert()

with open(convert_model_path, "wb") as f:
    f.write(tflite_model)