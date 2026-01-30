import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU

np.random.seed(1)

NUM_OF_CLASSES = 3

NUM_SAMPLES = 1000
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Example dataset generation
input_images = np.random.rand(NUM_SAMPLES, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
masks = np.random.randint(0, NUM_OF_CLASSES, size=(NUM_SAMPLES, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.int32)

class TestMeanIou(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

# Example dataset generation function
def generate_dataset(num_samples):
    for i in range(num_samples):
        input_image = input_images[i]
        mask = masks[i]

        yield input_image, mask

def from_generator():
    print("\ntf.data.Dataset.from_generator")

    dataset = tf.data.Dataset.from_generator(generate_dataset, args=[NUM_SAMPLES],
                    output_types=(tf.float32, tf.int32, tf.float32))
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def process_data(input_image, mask):
    # example process function
    return input_image, mask

def from_numpy_array():
    print("\ntf.data.Dataset.from_tensor_slices")

    dataset = tf.data.Dataset.from_tensor_slices((input_images, masks))
    dataset = dataset.map(process_data)
    dataset = dataset.shuffle(NUM_SAMPLES).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__=='__main__':
    input_image = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    output = Conv2D(NUM_OF_CLASSES, 1, activation='softmax', name='output')(input_image)

    model = Model(inputs=input_image, outputs=[output])
    model.compile(optimizer='adam',
                  loss={'output': 'sparse_categorical_crossentropy'},
                  metrics={'output': [TestMeanIou(num_classes=NUM_OF_CLASSES)]})
    model.summary()


    '''
    tf.data.Dataset.from_tensor_slices
    This works
    '''
    dataset = from_numpy_array()

    model.fit(dataset, epochs=2)    # train


    '''
    tf.data.Dataset.from_generator
    This results in following error -

    File "./venv/lib/python3.8/site-packages/keras/src/metrics/iou_metrics.py", line 123, in update_state  **
        if y_true.shape.ndims > 1:

    TypeError: '>' not supported between instances of 'NoneType' and 'int'
    '''
    # get tf dataset
    dataset = from_generator()

    model.fit(dataset, epochs=2)    # train

class TestMeanIou(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        print("y_true", type(y_true), y_true)
        print("y_pred", type(y_pred), y_pred)
        print("sample_weight", type(sample_weight), sample_weight)

        y_true = tf.cast(y_true, tf.int32)
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

def from_generator():
    print("\ntf.data.Dataset.from_generator")

    dataset = tf.data.Dataset.from_generator(generate_dataset, args=[NUM_SAMPLES],
                    output_types=(tf.float32, tf.int32),
                    output_shapes=((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), (IMAGE_SIZE[0], IMAGE_SIZE[1]))
                    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)