# tf.random.uniform((None,), dtype=tf.string) ← Input is a 1D tensor of strings representing image paths

import tensorflow as tf
import cv2
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, base_model=None):
        super().__init__()
        # If a base_model is provided, use it; otherwise create a dummy model for demonstration
        if base_model is None:
            # Simple CNN on 28x28 grayscale images
            inputs = tf.keras.Input(shape=(28, 28, 1))
            x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(10, activation='softmax')(x)  # Assume 10 classes
            self.base_model = tf.keras.Model(inputs, outputs)
        else:
            self.base_model = base_model

    def preprocess_image_tensor(self, image_path_tensor):
        """
        Preprocess a single image path string tensor into normalized grayscale 28x28 image tensor.
        This uses tf.py_function to wrap OpenCV numpy-based preprocessing inside TensorFlow graph.

        Args:
          image_path_tensor: scalar tf.string tensor

        Returns:
          4D float32 tensor (1, 28, 28, 1), normalized to [0,1]
        """
        def _preprocess(path_str):
            # path_str is a numpy bytes array, e.g. b'foo.png'
            path = path_str.decode('utf-8')
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28)).astype('float32')
            img /= 255.0
            img = np.expand_dims(img, axis=(0, -1))  # Shape (1, 28, 28, 1)
            return img

        img_tensor = tf.py_function(func=_preprocess,
                                    inp=[image_path_tensor],
                                    Tout=tf.float32)
        # Set shape explicitly since tf.py_function loses shape info
        img_tensor.set_shape([1, 28, 28, 1])
        return img_tensor

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serving(self, image_paths):
        """
        Takes a batch of image path strings, runs preprocessing and base model inference,
        and returns class indices per image.

        Args:
          image_paths: 1D tf.string tensor - batch of image file paths

        Returns:
          dict with key "class_index" and value int32 tensor of shape (batch_size,)
        """
        batch_images = []
        for i in tf.range(tf.shape(image_paths)[0]):
            img = self.preprocess_image_tensor(image_paths[i])
            batch_images.append(img)
        # Concatenate to get batch tensor of shape (batch_size, 28, 28, 1)
        input_batch = tf.concat(batch_images, axis=0)
        probabilities = self.base_model(input_batch, training=False)
        class_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
        return {"class_index": class_ids}


def my_model_function():
    # Instantiate MyModel with no external base model for demonstration
    return MyModel()

def GetInput():
    # Returns a batch of 2 dummy image path strings as tf.Tensor compatible with MyModel.serving
    # For demonstration, these paths won't actually load images unless you replace with real paths.
    # But this matches the expected input signature: shape = (batch_size,), dtype=tf.string
    image_paths = tf.constant(["sample_image1.png", "sample_image2.png"], dtype=tf.string)
    return image_paths

