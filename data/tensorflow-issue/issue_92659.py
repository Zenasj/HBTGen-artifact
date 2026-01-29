# tf.random.uniform((4, 32, 230, 288, 1), dtype=tf.float32) ← inferred input shape from model inputs vxm_dense_source_input and vxm_dense_target_input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Since the original model is Voxelmorph's VxmDense,
        # and direct code or weights unavailable,
        # we'll mock a simplified version with two inputs: moving and fixed,
        # producing two outputs as Identity ops did in original frozen graph.

        # Voxelmorph-like feature extractor layers (placeholders)
        self.conv3d_1 = tf.keras.layers.Conv3D(16, kernel_size=3, padding='same', activation='relu')
        self.conv3d_2 = tf.keras.layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')
        # Simple flow field output as per registration models - placeholder
        self.flow_field = tf.keras.layers.Conv3D(3, kernel_size=3, padding='same', activation=None)
        # Warp layer might be required, but here used dummy Identity-like output

    def call(self, inputs, training=False):
        # inputs expected as list or tuple: [moving, fixed]
        moving, fixed = inputs

        # Feature extraction from moving
        fmv = self.conv3d_1(moving)
        fmv = self.conv3d_2(fmv)

        # Feature extraction from fixed
        ffx = self.conv3d_1(fixed)  # share conv weights
        ffx = self.conv3d_2(ffx)

        # Simple difference feature
        diff = fmv - ffx

        # Estimating flow field (registration vector field)
        flow = self.flow_field(diff)

        # As Voxelmorph often outputs warped moving image and flow,
        # here we mimic two outputs
        # Output 1: flow field tensor
        # Output 2: warped moving (simulated as moving + flow) as a placeholder
        
        warped_moving = moving + tf.pad(flow, [[0,0],[0,0],[0,0],[0,0],[0, 1]])[:, :, :, :, :1] 
        # pad flow channels to match moving last dim (1 channel), simplified

        return flow, warped_moving


def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weight loading because original .h5 unavailable,
    # but this is a functional placeholder model reflecting original inputs and outputs.
    return MyModel()


def GetInput():
    # Return a tuple of random tensor inputs matching model expected shape:
    # moving and fixed images of shape (4, 32, 230, 288, 1), dtype tf.float32 as per original input specs.

    moving = tf.random.uniform(shape=(4, 32, 230, 288, 1), dtype=tf.float32)
    fixed = tf.random.uniform(shape=(4, 32, 230, 288, 1), dtype=tf.float32)

    return (moving, fixed)

