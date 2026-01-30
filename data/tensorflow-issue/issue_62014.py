from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_model_optimization as tfmot

inp1 = tf.keras.Input(shape=[2,4,8], batch_size = 1,name = 'input1')
inp2 = tf.keras.Input(shape=[2,4,8], batch_size = 1,name = 'input2')
r1 =tf.keras.layers.ReLU()(inp1)
r2 = tf.keras.layers.ReLU()(inp2)
c1 = tf.keras.layers.Concatenate(axis = -1)([r1,r2])

scheme_16_8 = tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(
    disable_per_axis=False, num_bits_weight=8, num_bits_activation=16)

test_model = tf.keras.Model(inputs=[inp1, inp2], outputs=c1)
annotated_model = tf.keras.models.clone_model(
        test_model,      
    )
ann_model = tfmot.quantization.keras.quantize_annotate_model(annotated_model)
q_model = tfmot.quantization.keras.quantize_apply(ann_model, scheme = scheme_16_8)

converter = tf.lite.TFLiteConverter.from_keras_model(q_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]#tf.lite.OpsSet.TFLITE_BUILTINS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
quantized_tflite_model = converter.convert()

q_model_2 = tfmot.quantization.keras.quantize_apply(ann_model)

converter = tf.lite.TFLiteConverter.from_keras_model(q_model2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]#tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
quantized_tflite_model2 = converter.convert()