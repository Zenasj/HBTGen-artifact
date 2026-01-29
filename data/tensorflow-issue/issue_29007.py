# tf.random.uniform((B, H, W, C), dtype=tf.float32)
import tensorflow as tf
import random

class MyModel(tf.keras.Model):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.count = tf.Variable(0, trainable=False, dtype=tf.int32)
        # We store images in a Python list, since dynamic list manipulations are tricky in TF.
        # For TF 2.x eager execution compatibility,
        # we keep images as a Python list of Tensors.
        # This matches the original ImagePool intent.
        self.images = []

    def call(self, images):
        # Emulate the 'query' method behavior:
        # Given a list or batch of images (tensors),
        # return_images is formed by either buffering,
        # or returning cached images with 50% chance.

        # For ease, we require images to be a list of tensors.
        return self.query(images)

    def query(self, images):
        # If pool_size == 0, just return input images
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            if self.count.numpy() < self.pool_size:
                # Buffer not full, insert current image and return it
                self.images.append(image)
                self.count.assign_add(1)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # 50% chance to return a previously stored image
                    random_id = random.randint(0, self.pool_size - 1)
                    # Clone the tensor to simulate the original clone behavior
                    # Since tf.Tensors are immutable, cloning means just return a new tensor with same values
                    tmp = tf.identity(self.images[random_id])
                    # Replace stored image with current image
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # 50% chance return current image as is
                    return_images.append(image)
        return return_images


def my_model_function():
    # For demonstration, set pool_size to 50 (arbitrary)
    return MyModel(pool_size=50)


def GetInput():
    # Return a list of input images matching the expected input of MyModel.query/call
    # Assumption:
    # - Each image tensor shape: (64, 64, 3) (e.g., typical RGB image)
    # - Batch size: 10 images
    batch_size = 10
    H, W, C = 64, 64, 3

    # Generate a list of 10 random image tensors
    images = [tf.random.uniform((H, W, C), dtype=tf.float32) for _ in range(batch_size)]
    return images

