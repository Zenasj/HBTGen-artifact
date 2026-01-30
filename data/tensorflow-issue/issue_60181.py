from tensorflow import keras

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4", output_shape=(1001,))])
m.build([None, 224, 224, 3])# Batch input shape.
images = '/home/Downloads/ILSVRC2012_img_val'
classes = '/home/Documents/ImageNetLabels.txt'
labels ='/home/Documents/val.txt'

with open(labels, 'r') as f:
	label_name = [line.strip() for line in f.readlines()]

class_map = {}
with open(classes, 'r') as f:
	classes = [line.strip() for line in f]
	for i, class_name in enumerate(classes):
		class_map[class_name] = i

test_labels=[]
for label in label_name:
	if label in class_map:
		test_labels.append(class_map[label])
	else:
		print(f"label '{label} not found in class_map")

image_paths = [os.path.join(images, filename) for filename in os.listdir(images)]
dataset = tf.data.Dataset.from_tensor_slices((image_paths, test_labels))
def preprocess_image(image_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, [224,224])
	image = tf.image.convert_image_dtype(image, tf.float32)
	image /= 255.0
	return image

dataset = dataset.map(lambda image_path, label: (preprocess_image(image_path), label))

dataset = dataset.batch(batch_size=32)
m.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loss, accuracy = m.evaluate(dataset)
print('loss: ', loss)
print('accuracy: ', accuracy)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4", output_shape=(1001,))])
m.build([None, 224, 224, 3]) # Batch input shape.

images = '/home/Downloads/ILSVRC2012_img_val/'
classes = '/home/Documents/tflite_models/imagenet_classes.txt'
labels ='/home/Documents/val.txt'
#labels ={}

with open(labels, 'r') as f:
	label_name = [line.strip() for line in f.readlines()]

class_map = {}
with open(classes, 'r') as f:
	classes = [line.strip() for line in f]
	for i, class_name in enumerate(classes):
		class_map[class_name] = i

test_labels=[]
for label in label_name:
	if label in class_map:
		test_labels.append(class_map[label])
	else:
		print(f"label '{label} not found in class_map")

#print(type(test_labels))
test_labels = tf.one_hot(test_labels, 1001)

image_paths = [os.path.join(images, filename) for filename in os.listdir(images)]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, test_labels))

def preprocess_image(image_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, [224,224])
	#image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.cast(image, tf.float32) / 255.0
	return image

dataset = dataset.map(lambda image_path, label: (preprocess_image(image_path), label))
dataset = dataset.batch(batch_size=32)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loss, accuracy = model.evaluate(dataset)

print('loss: ', loss)
print('accuracy: ', accuracy)