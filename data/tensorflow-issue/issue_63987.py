# tf.random.normal((1, 64, 64, 3), dtype=tf.float32) ← Inferred input shape from convolutional model example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the two sub-models discussed in the issue for fusion

        # Model A: Fully connected deep model from chunk 2
        self.model_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Model B: Convolutional model from chunk 3 example (Conv2D + DepthwiseConv2D)
        self.model_conv = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(64, (1, 1), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

    def call(self, inputs, training=False):
        # The model expects a single input tensor compatible with conv model (checking shape)
        # If input shape matches FC model input, reshape or adapt
        # Here, we assume inputs shaped for conv model: [B,64,64,3]

        # Pass through convolutional model
        conv_out = self.model_conv(inputs, training=training)  # Shape: (B,10)

        # Prepare input for FC model: flatten spatial dims & channels as vector
        # Using a simple reshape: flatten inputs per batch to vectors -> shape (B, 64*64*3)
        flat_inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])

        # Pass through FC model
        fc_out = self.model_fc(flat_inputs, training=training)  # Shape: (B,1)

        # Fuse outputs by comparison: Here just check shape compatibility first
        # Implement a loose comparison logic:
        # We can compare predictions after mapping to similar dimension or just output both results

        # For demonstration, compute difference in mean predictions after reducing conv_out (last dim)
        conv_pred = tf.reduce_mean(conv_out, axis=1, keepdims=True)  # (B,1)

        # Compute absolute difference between conv_pred and fc_out
        diff = tf.abs(conv_pred - fc_out)  # (B,1)

        # Return a dict with both outputs and their difference
        return {
            'fc_output': fc_out,
            'conv_output': conv_out,
            'mean_conv_pred': conv_pred,
            'difference': diff
        }

def my_model_function():
    # Initialize and compile the fused model (compilation optional, but included for completeness)
    model = MyModel()
    # We compile the FC model directly for losses if needed
    # Compile only main model to avoid confusion (losses on fc_output typically)
    model.model_fc.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )
    # For conv model, typical categorical loss might be necessary but kept separate
    # Here no training code is required; just return the model
    return model

def GetInput():
    # Return a valid input tensor matching conv model input shape (which is our assumed primary input):
    # Batch size 1, image 64x64 with 3 channels
    # Use tf.random.normal as per example from chunk 3
    return tf.random.normal((1, 64, 64, 3), dtype=tf.float32)

