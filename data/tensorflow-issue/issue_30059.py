# tf.random.uniform((1, 192, 192, 3), dtype=tf.float32) ← inferred input shape from the example (batch=1, height=192, width=192, channels=3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, create a simple model mimicking the description:
        # A convolutional base ending in a conv layer named 'conv2d_33'
        # and some dense layers producing 2-class output as in the example.
        # This is a placeholder model structure inferred from the issue.
        
        # Conv base layers (simplified and reduced)
        self.conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        # This layer is named 'conv2d_33' to match last_conv_layer in the example
        self.conv_last = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_33')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv_last(x)
        self.feature_maps = x  # save for later gradient use
        x = self.pool(x)
        output = self.dense(x)
        return output
    
    @tf.function(jit_compile=True)
    def get_grad_cam(self, inputs, class_index):
        """
        Compute the gradient of the output for `class_index` w.r.t. the last conv layer output.
        Returns:
          feature_maps: last conv layer output
          grads: gradients of class output w.r.t. feature maps
        """
        with tf.GradientTape() as tape:
            tape.watch(self.feature_maps)
            preds = self.call(inputs)
            # Pick the class output we want
            class_output = preds[:, class_index]
        grads = tape.gradient(class_output, self.feature_maps)
        return self.feature_maps, grads

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor matching the input shape (batch, height, width, channels)
    # Using float32 type as in typical image models
    return tf.random.uniform((1, 192, 192, 3), dtype=tf.float32)

