# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← inferred typical input for VGG preprocess_input: batch of images with 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load VGG19 without the top layer, outputting block5_pool features as in the example
        base_vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        base_vgg.trainable = False
        outputs = [base_vgg.get_layer(name).output for name in ['block5_pool']]
        self.vgg = tf.keras.Model(inputs=base_vgg.input, outputs=outputs)
        
    def call(self, inputs, training=False):
        # inputs: batch of images (B, H, W, 3), float32 expected by preprocess_input
        # For demo, apply preprocessing inside the model using keras.applications.vgg19.preprocess_input
        # Note: preprocess_input expects float images with channels last, values in expected range
        
        # Preprocess inputs
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        
        # Pass to VGG network
        features = self.vgg(preprocessed, training=training)  # list of one tensor from block5_pool
        # features is a list with one tensor: shape (B, 7, 7, 512)
        features_tensor = features[0]
        
        # Flatten the features to (B, 7*7*512)
        flattened = tf.reshape(features_tensor, [tf.shape(features_tensor)[0], -1])
        
        return flattened

def my_model_function():
    # Return an instance of MyModel
    # Add a workaround initialization to avoid the tf.constant scope issue with preprocess_input
    # This initializes internal keras_applications globals outside tf.function scopes
    temp = tf.zeros([4, 32, 32, 3], dtype=tf.float32)
    tf.keras.applications.vgg19.preprocess_input(temp)
    return MyModel()

def GetInput():
    # Return a random float32 tensor input that matches expected VGG input shape (B, H, W, C)
    # Typical VGG input size is 224x224 with 3 channels, batch size can be arbitrary, say 8
    batch_size = 8
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=255, dtype=tf.float32)

