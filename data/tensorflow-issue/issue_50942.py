from tensorflow import keras

import tensorflow as tf
import model as modellib
import coco
import os 
import sys

# Enable eager execution
tf.compat.v1.enable_eager_execution()

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir='logs', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True)
model = model.keras_model

tf.saved_model.save(model, "tflite")

# Preparing before conversion - making the representative dataset
ROOT_DIR = os.path.abspath("../")
CARS = os.path.join(ROOT_DIR, 'Mask_RCNN\\mrcnn\\smallCar')

IMAGE_SIZE = 224
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(CARS)
    for i in range(100):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('modelQuantized.tflite', 'wb') as f:
  f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error

converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('modelQuantized.tflite', 'wb') as f:
  f.write(tflite_model)