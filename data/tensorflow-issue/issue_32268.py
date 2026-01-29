# tf.random.uniform((None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
import tensorflow as tf

# Assumptions and notes:
# - Input shape inferred from tf.keras.applications.NASNetMobile default input (IMG_SIZE, IMG_SIZE, 3).
#   IMG_SIZE is typically 224 or 256 for NASNetMobile, assuming 224 here.
# - num_labels taken from code as 229.
# - The core of the issue is that with eager enabled, the final Dense layer's activation behaves differently
#   creating output identity node vs sigmoid node. We model both behaviors as two submodels.
# - We fuse both models to compare their outputs (dense raw sigmoid activation output).
# - Forward pass returns a dictionary holding outputs of eager and non-eager versions.
# - This allows easy comparison downstream to inspect differences caused by eager mode.
# - Without full code for the exact input preprocessing, we simply assume raw input of shape (None, IMG_SIZE, IMG_SIZE, 3).
# - For demonstration, both branches share weights, but differ slightly in the construction to reflect reported differences.
# - Note: tf.keras.applications.NASNetMobile requires input images scaled appropriately (0-255 or normalized).
# - We use activation='sigmoid' on output layer as in original example.
# - The fused model shows a plausible way to examine the difference in outputs that was the issue focus.

IMG_SIZE = 224
num_labels = 229

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Shared base NASNetMobile model (frozen weights)
        self.base_model = tf.keras.applications.NASNetMobile(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Output Dense layer with sigmoid activation
        self.output_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid', name="cinemanet_output")

        # For simulating different eager vs non-eager behavior,
        # create two "submodel" branches applying same layers but
        # will differ in output node naming or activation wrapping as per issue.

        # Eager branch - builds model normally, yielding Identity as output op.
        self.eager_dense = self.output_layer  # reuse the same layer here

        # Non-eager branch - mimic non-eager construction where activation might be separate.
        # Simulate by splitting activation for explicit sigmoid call to differentiate output op.
        # We will create a separate Dense without activation and then sigmoid as Lambda.
        self.non_eager_dense_no_activation = tf.keras.layers.Dense(num_labels, activation=None, name="cinemanet_output_no_activation")
        # Initialize weights of dense layer same as original Dense for fair comparison
        # This will be done in build() once weights available, see below.
        self.non_eager_sigmoid = tf.keras.layers.Activation('sigmoid', name="cinemanet_output_sigmoid")

        # Flag to check if weights are copied from eager branch to non-eager branch
        self._weights_copied = False

    def build(self, input_shape):
        # Build base_model and global pooling layer if not already built
        dummy = tf.zeros(input_shape)
        _ = self.base_model(dummy)
        _ = self.global_pool(self.base_model.output)
        # Build output layers
        self.output_layer.build((input_shape[0], 1056))  # NASNetMobile output last dimension 1056
        self.non_eager_dense_no_activation.build((input_shape[0], 1056))
        self.non_eager_sigmoid.build((input_shape[0], num_labels))
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Base feature extraction
        features = self.base_model(inputs, training=training)
        pooled = self.global_pool(features)

        # Eager branch: dense with sigmoid activation inline
        eager_out = self.eager_dense(pooled, training=training)

        # Non-eager branch: dense without activation + separate sigmoid activation
        # Copy weights from eager_dense dense layer if not done yet
        if not self._weights_copied:
            self.non_eager_dense_no_activation.set_weights(self.eager_dense.get_weights())
            self._weights_copied = True
        non_eager_logits = self.non_eager_dense_no_activation(pooled, training=training)
        non_eager_out = self.non_eager_sigmoid(non_eager_logits)

        # Return dictionary with both outputs to enable inspection/comparison
        return {
            'eager_output': eager_out,
            'non_eager_output': non_eager_out
        }

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random float tensor matching the NASNetMobile input shape (batch size 1)
    # Using values in [0,1) normalized float32 as placeholder
    return tf.random.uniform((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)

