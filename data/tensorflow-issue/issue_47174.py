# tf.random.uniform((32, 32, 32, 3), dtype=tf.float32) ← inferred input shape (batch size arbitrary, 32x32 RGB images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape is (32,32,3) as per EfficientNetB0 example
        
        # Load EfficientNetB0 base model without top layers, weights pretrained on ImageNet
        # This model uses DepthwiseConv2D layers internally, source of nondeterminism on GPU.
        self.base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3)
        )
        
        # Some layers from the original snippet
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten_layer = tf.keras.layers.Flatten()
        
        # The dense layer at the end with softmax activation and fixed initializer for reproducibility
        initializer = tf.keras.initializers.GlorotUniform(seed=1)
        self.dense_layer = tf.keras.layers.Dense(
            10,
            use_bias=False,
            kernel_initializer=initializer,
            activation='softmax',
            name='Bottleneck'
        )
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)  # pass training to handle batch norm mode properly
        x = self.global_average_layer(x)
        x = self.flatten_layer(x)
        output = self.dense_layer(x)
        return output


def my_model_function():
    # Return instance of MyModel, freshly initialized with ImageNet weights on EfficientNetB0 backbone
    return MyModel()


def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # Batch size chosen arbitrarily as 32 following the example batch size in the issue
    return tf.random.uniform(shape=(32, 32, 32, 3), dtype=tf.float32)

