# tf.random.normal((32, 128, 128, 3), dtype=tf.float32), tf.random.uniform((32, 100, 4), dtype=tf.float32)
import tensorflow as tf

IMG_SIZE = 128
NUM_BOXES = 100
CROP_SIZE = 28
NB_DATA_SAMPLES = 32
BATCH_SIZE = 1

class CropLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        images_, boxes_ = inputs
        # Compute the box indices for crop_and_resize:
        # For each image in the batch, repeat the index NUM_BOXES times
        batch_size = tf.shape(images_)[0]
        box_indices = tf.reshape(
            tf.repeat(
                tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=-1),
                NUM_BOXES,
                axis=-1
            ),
            shape=(-1,)
        )
        # Reshape boxes to (batch_size * NUM_BOXES, 4)
        boxes_reshaped = tf.reshape(boxes_, (-1, 4))
        # crop_and_resize returns (batch_size * NUM_BOXES, CROP_SIZE, CROP_SIZE, 3)
        cropped_images = tf.image.crop_and_resize(
            image=images_,
            boxes=boxes_reshaped,
            box_indices=box_indices,
            crop_size=(CROP_SIZE, CROP_SIZE)
        )
        return cropped_images

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.crop_layer = CropLayer()

    def call(self, inputs, **kwargs):
        """
        inputs: tuple or list with two tensors
          - images: shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
          - boxes: shape (BATCH_SIZE, NUM_BOXES, 4)
        Returns:
          cropped images of shape (BATCH_SIZE * NUM_BOXES, CROP_SIZE, CROP_SIZE, 3)
        """
        images, boxes = inputs
        return self.crop_layer([images, boxes])

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple (images, boxes) matching the inputs used for MyModel
    images = tf.random.normal(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    boxes = tf.random.uniform(shape=(BATCH_SIZE, NUM_BOXES, 4), minval=0, maxval=1, dtype=tf.float32)
    return (images, boxes)

