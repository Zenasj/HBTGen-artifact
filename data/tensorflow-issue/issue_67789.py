import numpy as np

import tensorflow as tf

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        # Add other features here if necessary
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224])  # Adjust size as necessary
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1] if required
    return image

def load_tfrecord_dataset(tfrecord_path, batch_size=1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size)
    return dataset


def representative_dataset(tfrecord_path, num_samples):
    dataset = load_tfrecord_dataset(tfrecord_path)
    for data in dataset.take(num_samples):
        yield [data]


# Load the TensorFlow SavedModel
saved_model_dir = 'model_weights/tflite/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Set optimization to default for INT8 conversion
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set the representative dataset
tfrecord_path = '/home/ai_server/Shabbir/Hand_Keypoint_Detection/data/coco_testdev.record-00001-of-00050'
num_samples = 100  # Adjust the number of samples as needed
converter.representative_dataset = lambda: representative_dataset(tfrecord_path, num_samples)

# Ensure that input and output tensors are quantized
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.allow_custom_ops = True
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open('centernet_int8_03.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# Load the TFLite model and allocate tensors.
# model_path = "workspace/tflite/model_6_May.tflite"
model_path = "/home/ai_server/Shabbir/DISIGN/model_weights/centernet_int8_03.tflite"
label_map_path = '/home/ai_server/Shabbir/DISIGN/model_weights/label_map.pbtxt'
image_path = '/home/ai_server/Shabbir/Hand_Keypoint_Detection/test/806.jpg'
# image_path = '/home/shabbirmarfatiya/Shabbir/Project/ML_Tasks/Hand_Gesture_Recognition/Hand_Keypoint_Detection/FreiHAND_Dataset/test/'+str(dir_list[15])

# Initialize TensorFlow Lite Interpreter.object_detection.utils import label_map_util
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Label map can be used to figure out what class ID maps to what
# label. `label_map.txt` is human-readable.
# category_index = {1: {'id': 1, 'name': 'person'}}
category_index = label_map_util.create_category_index_from_labelmap(
    label_map_path)

# print(category_index)

label_id_offset = 1

image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
image_numpy = image.numpy()
print(image_numpy.shape)

input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
# Note that CenterNet doesn't require any pre-processing except resizing to the
# input size that the TensorFlow Lite Interpreter was generated with.
input_tensor = tf.image.resize(input_tensor, (224, 224))
(boxes, classes, scores, num_detections, kpts, kpts_scores) = detect(interpreter, input_tensor,include_keypoint=True)
print("kpts:",scores[0])
print("kpts_scores:",kpts[0][0]*image_numpy.shape[1])
print("Boxes:", boxes)
print("classes:", classes)
print("num_detections:", num_detections)
# print("kpts_scores:",kpts[0][0])
vis_image = plot_detections(
    image_numpy[0],
    boxes[0],
    classes[0].astype(np.uint32) + label_id_offset,
    scores[0],
    category_index,
    keypoints=kpts[0],
    keypoint_scores=kpts_scores[0])

plt.figure(figsize = (15, 10))
plt.imshow(vis_image)

py
import torch
import torchvision
import ai_edge_torch

mobilenet_model = torchvision.models.mobilenet_v3_small()
sample_inputs = (torch.randn(1, 3, 224, 224),)

edge_model = ai_edge_torch.convert(mobilenet_model.eval(), sample_inputs)
edge_model.export("mobilenet_v3_small.tflite")