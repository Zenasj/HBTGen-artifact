# tf.random.uniform((8, 10, 10, 3), dtype=tf.uint8) ← Inferred input is batch of 8 images 10x10x3 uint8, encoded as JPEG strings

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        # Initialize a Conv2D layer inside the strategy scope
        with self.strategy.scope():
            self.conv = layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')
        self.parallel_iterations = 10

        # Prepare encoded image dataset
        H, W, C = 10, 10, 3
        # Generate a batch of 8 zero uint8 images and encode each as JPEG string tensor
        imgs = tf.cast(tf.zeros([8, H, W, C]), tf.uint8)
        encodes = []
        for img in imgs:
            encode = tf.io.encode_jpeg(img)
            encodes.append(encode)
        encodes = tf.stack(encodes, axis=0)  # Shape (8,) of string tensors

        # Create tf.data.Dataset from encoded JPEG strings, batch size 2, distributed dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(encodes)
        self.dataset = self.dataset.batch(2)
        self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)

    def preprocess(self, encoded):
        # Decode JPEG string into uint8 image tensor [H,W,C]
        image = tf.io.decode_jpeg(encoded, channels=3)
        # Resize to 20x20 float32 tensor (default dtype of resize)
        image = tf.image.resize(image, [20, 20])
        return image

    @tf.function
    def serving(self, inputs):
        # inputs: batch of encoded JPEG strings, shape (batch_size,)
        # Apply preprocessing across the batch in graph mode with tf.map_fn
        image = tf.map_fn(
            self.preprocess,
            inputs,
            fn_output_signature=tf.float32,
            parallel_iterations=self.parallel_iterations,
        )
        # Forward pass through conv layer
        prediction = self.conv(image)
        return prediction

    @tf.function
    def infer(self, serve_summary_writer):
        # Performs inference over distributed dataset and writes summary
        batch = tf.cast(0, tf.int64)
        for data in self.dataset:
            # Run serving step on distributed dataset batch using the strategy
            prediction_per_replica = self.strategy.run(self.serving, args=(data,))
            # Each replica returns a tensor; collect list of these tensors
            prediction_tensor_list = prediction_per_replica.values
            # Concatenate predictions along batch dimension
            prediction_concat = tf.concat(prediction_tensor_list, axis=0)
            # Write summary of prediction tensor for current step
            tf.summary.write(tag="prediction", tensor=prediction_concat, step=batch)
            batch += 1

def my_model_function():
    # Create a MirroredStrategy and return the MyModel instance with it
    strategy = tf.distribute.MirroredStrategy()
    return MyModel(strategy)

def GetInput():
    # Generate a tensor of encoded JPEG images matching input expected by MyModel.serving
    # Inputs should be a batch of JPEG encoded strings, shape (batch_size,)
    batch_size = 8
    H, W, C = 10, 10, 3
    # Create zero uint8 images and encode to JPEG strings
    imgs = tf.cast(tf.zeros([batch_size, H, W, C]), tf.uint8)
    encodes = []
    for img in imgs:
        encode = tf.io.encode_jpeg(img)
        encodes.append(encode)
    encodes = tf.stack(encodes, axis=0)  # Tensor of shape (batch_size,) of string dtype
    return encodes

