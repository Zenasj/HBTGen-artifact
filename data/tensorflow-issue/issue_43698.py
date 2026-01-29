# tf.random.uniform((10, 8, 4, 6, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, nframe=10):
        super(MyModel, self).__init__()
        self.nframe = nframe
        # Example conv layer - not actively used here but from original snippet
        self.conv_first = tf.keras.layers.Conv2D(4, (3, 3))
    
    def call(self, x):
        # Use a fixed size TensorArray to ensure correct shape inference in AutoGraph mode
        aligned_fea = tf.TensorArray(dtype=tf.float32, size=self.nframe)
        
        def cond(i, N, fea_col):
            return i < N
        
        def body(i, N, fea_col):
            # Write the input tensor x into TensorArray at index i
            fea_col = fea_col.write(i, x)
            i = tf.add(i, 1)
            return i, N, fea_col
        
        _, _, aligned_fea = tf.while_loop(cond, body, [0, self.nframe, aligned_fea])
        
        tf.print("aligned_fea size:", aligned_fea.size())
        t = aligned_fea.stack()
        tf.print("Stacked tensor shape:", t.shape)
        
        # Because we used fixed size TensorArray, shape inference works and t's first dimension is nframe
        # Output stacked tensor directly, shape should be [nframe, 8,4,6,3]
        return t

def my_model_function():
    # Initialize MyModel with default nframe=10
    return MyModel()

def GetInput():
    # Input tensor matching the expected shape put into TensorArray: [8, 4, 6, 3]
    return tf.random.uniform((8, 4, 6, 3), dtype=tf.float32)

