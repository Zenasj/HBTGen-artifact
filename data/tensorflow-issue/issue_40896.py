# tf.random.uniform((B, None, None, None, 1), dtype=tf.float32)
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CustomLayer, self).__init__(name=name, **kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, inputs):
        # inputs shape: (batch_size, height, width, channels)
        output_1 = self.conv_1(inputs)
        output_2 = self.conv_2(inputs)
        return output_1, output_2

    def compute_output_shape(self, input_shape):
        output_shape_1 = self.conv_1.compute_output_shape(input_shape)
        output_shape_2 = self.conv_2.compute_output_shape(input_shape)
        return output_shape_1, output_shape_2

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Wrap the CustomLayer inside a manual TimeDistributed logic because
        # original TimeDistributed does not support layers with multiple outputs in TF 2.2.
        self.custom_layer = CustomLayer()

    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, height, width, channels)
        # Manually apply CustomLayer to each timestep (simulate TimeDistributed)
        # instead of using tf.keras.layers.TimeDistributed directly to avoid the TF 2.2 limitation.
        # TensorFlow 2.2 TimeDistributed does not support multiple outputs.
        
        # We will use tf.vectorized_map or tf.map_fn for batch wise and timestep wise mapping over inputs:
        # Since inputs are 5D: (B, T, H, W, C), we map over axis=1 (timesteps)
        
        # Define a function that applies CustomLayer to a single timestep tensor: (B, H, W, C)
        def apply_layer_on_timesteps(timestep_input):
            # timestep_input shape: (B, H, W, C)
            # Actually map custom_layer on this tensor batchwise directly:
            # CustomLayer expects 4D tensor of shape (batch, H, W, C), so timestep_input
            # already has batch dimension. So for batches we just apply custom_layer directly.

            # Actually timestep_input shape inside tf.map_fn is (H, W, C), so
            # to preserve batch dimension, best to use tf.vectorized_map on axis=1.
            # However, typical input shape is (B, T, H, W, C).
            # tf.vectorized_map maps over the outermost dimension.
            # To vectorize over time dimension, we need to swap axes.

            # Because vectorized_map operates on 1D input, we will implement call with tf.map_fn,
            # mapping over timesteps on batch dimension separately. Safer to unroll.

            # Since we will map with tf.map_fn over time dimension for each batch,
            # timestep_input shape would be (H, W, C) (having removed batch dim)
            # but CustomLayer expects (B, H, W, C), so we must handle batch dimension carefully.

            # To stay simple and compatible with tf.function and jit_compile, 
            # let's implement manual batch and timestep loops with tf.vectorized_map and tf.map_fn.

            # To work around complexity, reshape inputs to merge batch and time dims, apply layer, then reshape back.

            raise NotImplementedError("apply_layer_on_timesteps should not be used directly")

        # Solution:
        # merge batch and time dims: (B*T, H, W, C)
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        height = tf.shape(inputs)[2]
        width = tf.shape(inputs)[3]
        channels = tf.shape(inputs)[4]

        inputs_reshaped = tf.reshape(inputs, (batch_size * time_steps, height, width, channels))
        output_1, output_2 = self.custom_layer(inputs_reshaped)

        # output shapes: (B*T, H_o, W_o, filters)
        # reshape back to (B, T, H_o, W_o, filters)
        output_1_shape = tf.shape(output_1)
        output_2_shape = tf.shape(output_2)

        out1 = tf.reshape(output_1,
                          (batch_size, time_steps,
                           output_1_shape[1],
                           output_1_shape[2],
                           output_1_shape[3]))
        out2 = tf.reshape(output_2,
                          (batch_size, time_steps,
                           output_2_shape[1],
                           output_2_shape[2],
                           output_2_shape[3]))
        return out1, out2

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input shape: (batch_size, timesteps, height, width, channels)
    # From example, channels=1.
    # Height and width are left None in example, so choose small values for test (e.g., 8,8).
    batch_size = 2
    time_steps = 3
    height = 8
    width = 8
    channels = 1
    return tf.random.uniform((batch_size, time_steps, height, width, channels), dtype=tf.float32)

