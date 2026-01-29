# tf.random.uniform((B=1, H=8, W=8, C=1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize layers for the upsampling branch
        self.conv1_up = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3,
            kernel_initializer=tf.keras.initializers.Constant(1),
            name="conv_1_up"
        )
        self.conv2_up = tf.keras.layers.Conv2D(
            filters=1, kernel_size=5,
            kernel_initializer=tf.keras.initializers.Constant(1),
            name="conv_2_up"
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), name="upsample")

        # Initialize layers for the transposed convolution branch
        self.conv1_tr = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3,
            kernel_initializer=tf.keras.initializers.Constant(1),
            name="conv_1_tr"
        )
        self.conv2_tr = tf.keras.layers.Conv2D(
            filters=1, kernel_size=5,
            kernel_initializer=tf.keras.initializers.Constant(1),
            name="conv_2_tr"
        )
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=2,
            kernel_initializer=tf.keras.initializers.Constant(1),
            strides=2,
            name="conv_transpose"
        )

    def call(self, inputs, training=False):
        # Upsample branch forward pass
        x_up = self.conv1_up(inputs)
        x_up = self.conv2_up(x_up)
        x_up = self.upsample(x_up)
        
        # Transposed convolution branch forward pass
        x_tr = self.conv1_tr(inputs)
        x_tr = self.conv2_tr(x_tr)
        x_tr = self.conv_transpose(x_tr)
        
        # Return both outputs for comparison or use externally
        return x_up, x_tr

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # From the issue and code, input shape is expected (1, 8, 8, 1) or resizable up to (1,16,16,1).
    # Provide shape (1, 8, 8, 1) as default input matching original setup.
    return tf.random.uniform(shape=(1, 8, 8, 1), dtype=tf.float32)

