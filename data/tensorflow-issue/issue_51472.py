# tf.random.uniform((BATCH_SIZE, 128, 128, 3), dtype=tf.float32) ← inferred input shape from custom generator and configurations

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(128, 128, 3), conv2d_filters=[32, 64, 128], embedding_nodes=64):
        super().__init__()
        self.embedding_nodes = embedding_nodes
        # Build the shared body model for embedding extraction
        self.body = self.build_body(input_shape, conv2d_filters)

    def build_body(self, input_shape, conv2d_filters):
        # Build a CNN body that extracts embeddings from input images
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for filters in conv2d_filters:
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.embedding_nodes)(x)
        # L2 normalize embeddings for triplet loss stability
        outputs = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
        return tf.keras.Model(inputs, outputs, name="EmbeddingBody")

    def call(self, inputs, training=False):
        """
        inputs: list or tuple of 3 tensors: [anchor, positive, negative]
        Each tensor shape: (batch_size, H, W, C)
        """
        anchor, positive, negative = inputs

        encoded_a = self.body(anchor, training=training)
        encoded_p = self.body(positive, training=training)
        encoded_n = self.body(negative, training=training)

        # Compute squared L2 distances along the embedding dim
        ap_distance = tf.reduce_sum(tf.square(encoded_a - encoded_p), axis=-1, keepdims=True)
        an_distance = tf.reduce_sum(tf.square(encoded_a - encoded_n), axis=-1, keepdims=True)

        # Return distances matching the original network output
        # This can feed into custom triplet loss function
        return ap_distance, an_distance

def my_model_function():
    # Using the default triplet network settings (input shape (128,128,3), conv filters example)
    # These can be changed based on actual configuration
    input_shape = (128, 128, 3)
    conv2d_filters = [32, 64, 128]
    embedding_nodes = 64
    model = MyModel(input_shape=input_shape, conv2d_filters=conv2d_filters, embedding_nodes=embedding_nodes)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # It expects a tuple/list of 3 input tensors: (anchor, positive, negative)
    batch_size = 16  # example batch size chosen reasonably
    height, width, channels = 128, 128, 3

    anchor = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    positive = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    negative = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    return [anchor, positive, negative]

