# tf.random.uniform((16, None, None, 1), dtype=tf.float32)  # Assuming (B=16, H=n/4, W=n/4, C=1) for input to MyModel

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model A: CNN operating on patches of shape (4, n/4, n/4, 1),
        # input batch size for A is 4 patches at a time.
        # We'll define a simple example CNN that outputs the same spatial shape with 1 channel.
        self.modelA = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='linear'),
        ])
        
        # Model B: CNN operating on full images of shape (1, n, n, 1),
        # outputs a vector of 10 classes.
        self.modelB = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ])
        
    def call(self, inputs):
        """
        inputs shape: (batch_size=16, H = n/4, W = n/4, C=1)
        Goal:
          - Split inputs into 4 groups of 4 patches (batch dimension = 16 = 4*4)
          - For each group of 4 patches (shape (4, H, W, 1)), run model A
          - Combine the 4 outputs from model A per group into a full image (batch size = 1, n, n, 1)
          - Run model B on that full image
          - batch dimension after grouping is 4 (since 16 patches /4 = 4 full images)
          - But original problem states model C input: (16, n/4, n/4, 1) and output (1, 10)
          - So at last combine outputs from model B to a single output (likely reduce mean) or
            just take first output as in example.
          
        Note: To avoid OOM when computing gradients, we process sequentially over groups, accumulating results.
        """
        batch_size = tf.shape(inputs)[0]
        # Number of groups of 4 patches:
        no_batches = batch_size // 4
        
        # Process patches in groups of 4:
        patch_outputs = []
        for i in tf.range(no_batches):
            patches = inputs[i*4:(i+1)*4]  # shape (4, H, W, 1)
            # Run model A on these 4 patches concurrently:
            patches_out = self.modelA(patches)  # shape (4, H, W, 1)
            patch_outputs.append(patches_out)
        
        # patch_outputs is a list of length no_batches, each (4, H, W, 1)
        # Stack to shape (no_batches, 4, H, W, 1)
        patches_stacked = tf.stack(patch_outputs)  # shape (no_batches, 4, H, W, 1)
        
        # Combine patches to reconstruct full images.
        # Assuming 4 patches correspond to 2x2 grid to form full image:
        # patches_stacked: (B_groups, 4, H, W, 1)
        # explicit combine function:
        
        def batch_combine(patches):
            # patches: (batch, 4, H, W, C)
            # Combine 4 patches per batch in a 2x2 grid
            # Output shape: (batch, 2*H, 2*W, C)
            top_row = tf.concat([patches[:,0], patches[:,1]], axis=2)  # concat width-wise
            bottom_row = tf.concat([patches[:,2], patches[:,3]], axis=2)
            full_images = tf.concat([top_row, bottom_row], axis=1)  # concat height-wise
            return full_images
        
        full_images = batch_combine(patches_stacked)  # shape: (no_batches, 2*H, 2*W, 1)
        
        # Run model B on full_images
        outputs_b = self.modelB(full_images)  # shape: (no_batches, 10)
        
        # According to user, model C output shape should be (1, 10) combining all
        # Here we reduce outputs_b by mean over batch to get (1, 10) tensor
        output = tf.reduce_mean(outputs_b, axis=0, keepdims=True)  # shape (1, 10)
        
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor matching input shape:
    # batch size 16, H = W = 32 (assumed n=128, so n/4=32 for demonstration), 1 channel
    # dtype float32
    batch_size = 16
    height = 32
    width = 32
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

