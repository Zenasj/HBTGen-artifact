from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(32, 32, 3)))
model.add(layers.Conv2D(16, (3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()

def to_list(x):
  if isinstance(x, list):
    return x
  return [x]

outs = []
for l in model.layers:
    output_tensors = [to_list(inbound.output_tensors)[0] for inbound in l.inbound_nodes]
    outs.append(output_tensors)

ins = []
for l in model.layers:
    input_tensors = [to_list(inbound.input_tensors)[0] for inbound in l.inbound_nodes]
    ins.append(input_tensors)

assert tf.executing_eagerly()
for i_tensor in ins:
    if i_tensor in outs:
        print("BOOM")