# tf.random.uniform((B, 1), dtype=tf.int32) ← Input shape inferred as a batch size dimension with single integer token (ID) per input

import tensorflow as tf

class GPUCompatibleEmbedding(tf.keras.layers.Embedding):
    @tf_utils_shape_type_conversion
    def build(self, input_shape):
        # Redefine build without CPU pinning to allow GPU placement
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

# Helper decorator from tf.keras.utils.tf_utils to decorate build correctly
def tf_utils_shape_type_conversion(func):
    # This decorator wraps shape and dtype conversion for tf_utils.shape_type_conversion 
    # because we can't import private tf_utils from tf.keras.utils here
    # Minimal no-op placeholder:
    return func


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the GPU compatible embedding layer workaround to avoid forced CPU placement
        self.embedding = GPUCompatibleEmbedding(input_dim=10, output_dim=1, input_shape=(1,))
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dense(x)
        return x


def my_model_function():
    # Initialize and return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random integer tensor input matching the expected embedding input shape: (batch_size, 1)
    # Here batch size is 16 as a typical example; values in 0-9 since input_dim=10
    return tf.random.uniform((16, 1), minval=0, maxval=10, dtype=tf.int32)

