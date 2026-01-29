# tf.random.uniform((B,), dtype=tf.string) ← Input is a batch of serialized tf.Example protos as tf.string tensors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, base_model, classes):
        super().__init__()
        self.base_model = base_model
        # classes: list of class strings, make a constant tensor for repeated use
        self.labels = tf.constant(classes)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, inputs):
        """
        inputs: a batch of serialized tf.Example strings, each containing a 'image' feature as a
        JPEG/PNG-encoded string.

        This function:
        - Parses each example to extract the image bytes,
        - Decodes and resizes each image to the base_model's expected input shape,
        - Calls the base_model on the preprocessed image batch,
        - Returns a dictionary with keys 'scores' (model outputs) and 'classes' (class labels repeated for batch).
        """

        def map_fn(example_proto):
            # Parse single example
            feature_spec = {'image': tf.io.FixedLenFeature([], tf.string)}
            features = tf.io.parse_single_example(example_proto, features=feature_spec)
            # Decode image (JPEG/PNG), convert to float32
            image = tf.io.decode_image(features['image'], channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            # Resize image with padding to model input height and width
            target_height, target_width = self.base_model.input_shape[1], self.base_model.input_shape[2]
            image = tf.image.resize_with_pad(image, target_height, target_width, method=tf.image.ResizeMethod.BILINEAR)
            return image

        # Map preprocessing over batch
        images = tf.map_fn(map_fn, elems=inputs, back_prop=False, dtype=tf.float32)

        # Run the base model on the batch
        logits = self.base_model(images)

        # Repeat classes labels for each batch element
        batch_size = tf.shape(logits)[0]
        repeated_labels = tf.repeat(tf.expand_dims(self.labels, 0), repeats=batch_size, axis=0)

        return {
            'scores': logits,
            'classes': repeated_labels
        }

    def call(self, inputs):
        # Optionally call base model directly
        return self.base_model(inputs)

def my_model_function():
    # Build a sample Keras base model similar to the example in the issue
    inputs = tf.keras.layers.Input(shape=(100, 100, 3))
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(2,2))(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    base_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Example class list; in practice replace with actual classes
    classes = ['class1', 'class2', 'class3']

    return MyModel(base_model, classes)

def GetInput():
    # Generate a batch of serialized tf.Example protos containing images as encoded bytes
    # Here for demonstration, we create dummy encoded images as empty PNG bytes,
    # but this needs to match the expected input for serve().

    def create_example(image_bytes):
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # We'll create a batch of 2 examples with blank PNG images (1x1 transparent PNG)
    blank_png_bytes = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\xdac\x00\x01\x00\x00\x05\x00\x01'
        b'\x0d\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    example1 = create_example(blank_png_bytes)
    example2 = create_example(blank_png_bytes)

    # Return as a tensor of dtype string and shape (2,), batch size 2
    return tf.constant([example1, example2], dtype=tf.string)

