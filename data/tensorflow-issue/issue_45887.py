import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def apply_quantization(layer):
    if isinstance(layer, keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = keras.models.clone_model(
    base_model, # base_model is a mobilenetv2 or resnet50
    clone_function=apply_quantization
)
q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.allow_custom_ops = True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.allow_custom_ops = True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# add this line before convert()
converter._experimental_new_quantizer = True

tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

MODEL_NAME = 'mobilenetv2'
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3), alpha=1.0, include_top=True, weights='imagenet',
    input_tensor=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
def apply_quantization(layer):
    if type(layer) == keras.layers.Conv2D:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = keras.models.clone_model(
    base_model,
    clone_function=apply_quantization
)

q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
# converter._experimental_new_quantizer = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
pathlib.Path('/tmp/tmp.tflite').write_bytes(tflite_model)

inputs = keras.layers.Input(shape=(224, 224, 3))
x1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x2 = keras.layers.Lambda(lambda x: tf.identity(x))(inputs)
x2 = tfmot.quantization.keras.quantize_annotate_layer(
    keras.layers.Conv2D(32, (3, 3), activation='relu')
)(x2)
x = keras.layers.concatenate(inputs = [x1,x2])
x = keras.layers.MaxPool2D([5,5], strides=5)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(2)(x)
base_model = keras.models.Model(inputs=inputs, outputs=x)
q_aware_model = tfmot.quantization.keras.quantize_apply(base_model)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter._experimental_new_quantizer = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
pathlib.Path('/tmp/tmp.tflite').write_bytes(tflite_model)