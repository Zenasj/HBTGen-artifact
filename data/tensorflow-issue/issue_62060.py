# tf.random.uniform((B, 5), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the dividers to slice the input tensor
        self.dividers = [0, 2, 5]
        # Sizes correspond to slice widths: [2, 3]
        self.sizes = [end - start for start, end in zip(self.dividers[:-1], self.dividers[1:] )]
        
        # Create Lambda layers to slice input into pieces
        # Store as a list of layers for submodules
        self.slice_layers = []
        for start, end in zip(self.dividers[:-1], self.dividers[1:]):
            # Each layer slices x[..., start:end]
            self.slice_layers.append(
                tf.keras.layers.Lambda(lambda x, s=start, e=end: x[..., s:e],
                                       name=f'slice_{start}_{end}')
            )

    def call(self, inputs):
        # Apply slice layers to input, produce list of sliced tensors
        outputs = [layer(inputs) for layer in self.slice_layers]
        return outputs

def my_model_function():
    # Return an instance of MyModel. This matches the original usage.
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel: (batch, channels)
    # Here channels = 5 based on dividers above.
    batch_size = 1  # reasonable default batch size
    channels = 5
    return tf.random.uniform((batch_size, channels), dtype=tf.float32)

