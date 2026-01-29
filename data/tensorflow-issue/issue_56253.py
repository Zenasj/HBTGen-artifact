# tf.random.uniform((1, 5, 5, 2), dtype=tf.float32) ← Input shape expected is (batch=1, height=5, width=5, channels=2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Dense layer to mirror the example architecture
        # Note: In the original code, the dense layer outputs 10 units, but the output used was 'x' before dense.
        # Here, to keep consistency with the core logic focusing on pre_process, we output the processed tensor.
        self.dense = tf.keras.layers.Dense(10)  # Although not used as output here, keeping consistent

    def call(self, inputs):
        # Equivalent of pre_process_shows_shape_range_meshgrid_BUG
        # This function caused issues in TF 2.3 but fixed in TF 2.8+.
        # It uses tf.shape to get dynamic shape, tf.range and tf.meshgrid to build positional grid for addition.
        
        # Extract dynamic height and width from input shape
        grid_x = tf.shape(inputs)[1]
        grid_y = tf.shape(inputs)[2]
        
        range_x = tf.range(grid_x)   # dynamic range along height
        range_y = tf.range(grid_y)   # dynamic range along width
        
        # meshgrid to create coordinate grids (notice order: meshgrid(y, x) to get [H, W])
        x_grid, y_grid = tf.meshgrid(range_y, range_x)   # shapes: (grid_x, grid_y)
        
        # Stack grids along last dimension to get shape (grid_x, grid_y, 2)
        b = tf.stack([y_grid, x_grid], axis=-1)
        b = tf.cast(b, tf.float32)
        
        # Add positional info to inputs
        y = inputs + b
        
        # Optionally, pass through a Dense layer (matching original sample code),
        # but since shape mismatch (Dense expects 2D input), skip or flatten first.
        # We'll skip here since original output was 'x' (i.e. preprocessed tensor).
        
        return y

def my_model_function():
    # Initialize and return model instance
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape (batch=1, height=5, width=5, channels=2)
    # Using uniform random values consistent with typical float32 input data for vision models.
    return tf.random.uniform((1, 5, 5, 2), dtype=tf.float32)

