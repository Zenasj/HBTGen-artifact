# tf.random.uniform((BATCH_SIZE, HEIGHT, WIDTH, 3), dtype=tf.uint8) ← inferred input shape for decoded images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using constants and config similar to Hyperparameters for this standalone class
        self.TFRECORDS_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.int64)
        }
        self.BATCH_SIZE = 32
        self.HEIGHT = 224
        self.WIDTH = 224
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def readTFRecs(self, dir_name):
        """
        Reads TFRecord files from directory and creates a TFRecordDataset
        and parses examples according to TFRECORDS_FORMAT.
        """
        # List files in directory
        TFRecFiles = tf.constant(tf.io.gfile.listdir(dir_name))
        # Prepend directory name to each filename
        TFRecFiles = tf.map_fn(lambda name: dir_name + '/' + name, TFRecFiles, fn_output_signature=tf.string)
        # Create dataset from TFRecord files
        TFRecDataset = tf.data.TFRecordDataset(TFRecFiles)
        # Save dataset length (not used directly here but kept from original design)
        self.dataset_len = tf.data.experimental.cardinality(TFRecDataset).numpy()
        # Parse example protos from TFRecordDataset
        # Using parse_single_example because entries are single Examples, not batches
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, self.TFRECORDS_FORMAT)
        Dataset = TFRecDataset.map(_parse_function, num_parallel_calls=self.AUTOTUNE)
        return Dataset

    @tf.function
    def decode_image(self, entry):
        """
        Decodes the image bytes in entry['image'] to a uint8 tensor with shape [H,W,3].
        Does not resize here, since original code did not resize at decode.
        """
        # Decode image raw bytes
        img = tf.image.decode_image(entry['image'], channels=3)
        # Set static shape [height, width, 3] if known or keep flexible
        img.set_shape([None, None, 3])
        # Optionally, resize to HEIGHT and WIDTH (inferred from Hyperparameters usage)
        img = tf.image.resize(img, [self.HEIGHT, self.WIDTH])
        # Convert to uint8 after resize (decode_image outputs uint8 or float?)
        img = tf.cast(img, tf.uint8)
        # Return a dict or tuple downstream; original code returned just image tensor
        # We'll return dict to match downstream input expectations
        # Keeping only 'image' key as decoded tensor to follow original intent
        return {'image': img, 'target': entry['target']}

    @tf.function
    def makeDataset(self, TFRecDataset):
        """
        Maps decode_image over TFRecDataset, shuffles, batches, and prefetches.
        Returns a tf.data.Dataset yielding batches of {'image': ..., 'target': ...}.
        
        This is adapted from the recommended working version in the issue (uncommented @tf.function).
        """
        Dataset = TFRecDataset.map(lambda entry: self.decode_image(entry), num_parallel_calls=self.AUTOTUNE)
        # Map to tuple of (image, one_hot_target) as commonly used for model input/output
        Dataset = Dataset.map(lambda entry: (entry['image'], tf.one_hot(entry['target'], 5)), num_parallel_calls=self.AUTOTUNE)
        Dataset = Dataset.shuffle(buffer_size=4000)
        Dataset = Dataset.batch(self.BATCH_SIZE)
        Dataset = Dataset.prefetch(tf.data.AUTOTUNE)
        return Dataset

    def call(self, inputs):
        """
        As this is primarily a DataParser and decoder class, no specific model architecture is given.
        To comply with tf.keras.Model subclassing and allow tf.function jit_compile, this 
        method accepts inputs and simply returns them or performs a dummy identity transform.
        
        This placeholder call aligns with usage where this class is used for dataset preparation.
        """
        # Identity pass-through as no model architecture specified
        return inputs


def my_model_function():
    """
    Instantiate and return the MyModel instance.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected decoded image input shape.
    Here we generate a batch of images (BATCH_SIZE, HEIGHT, WIDTH, 3) with uint8 pixels.
    """
    BATCH_SIZE = 32
    HEIGHT = 224
    WIDTH = 224
    # Random images in uint8 range 0-255
    return tf.random.uniform(
        shape=(BATCH_SIZE, HEIGHT, WIDTH, 3),
        minval=0,
        maxval=256,
        dtype=tf.int32
    ).numpy().astype('uint8')

