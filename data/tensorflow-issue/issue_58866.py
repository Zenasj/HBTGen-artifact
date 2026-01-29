# tf.random.uniform((B, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
# Assumptions for input shape:
# SEQUENCE_LENGTH = 10 (example)
# IMAGE_HEIGHT = 299 (InceptionV3 default)
# IMAGE_WIDTH = 299 (InceptionV3 default)
# CLASSES_LIST length = 5 (example number of classes for classification)

import tensorflow as tf

# Because of known issues with tensorflow-directml and AMD GPUs,
# it is recommended to use the tf.compat.v1.keras.layers.LSTM layer
# instead of tf.keras.layers.LSTM to avoid CudnnRNN CUDA ops being invoked.
# This class encapsulates that workaround by explicitly using the compat.v1 LSTM.

class MyModel(tf.keras.Model):
    def __init__(self,
                 sequence_length=10,
                 image_height=299,
                 image_width=299,
                 num_classes=5):
        super().__init__()
        
        # Use the TF Keras Applications InceptionV3 model without classifier top layers.
        self.inception_v3 = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(image_height, image_width, 3)
        )
        
        # Wrap the inception model in a TimeDistributed layer to apply to each frame in sequence.
        self.time_distributed_inception = tf.keras.layers.TimeDistributed(self.inception_v3)
        
        # Flatten the output per time step.
        self.time_distributed_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        
        # Use the `tf.compat.v1.keras.layers.LSTM` to avoid use of CudnnRNN op on AMD GPU
        self.lstm = tf.compat.v1.keras.layers.LSTM(32)
        
        # Final classification dense layer for number of classes
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        # inputs shape: (batch, sequence_length, image_height, image_width, 3)
        
        x = self.time_distributed_inception(inputs, training=training)
        x = self.time_distributed_flatten(x)
        x = self.lstm(x)
        out = self.classifier(x)
        return out

def my_model_function():
    # Return an instance of MyModel with a typical setup.
    # These dimension defaults can be adjusted as needed.
    return MyModel(sequence_length=10, image_height=299, image_width=299, num_classes=5)

def GetInput():
    # Generate random input tensor with appropriate shape and dtype.
    # Batch size chosen arbitrarily as 4 (like original code).
    batch_size = 4
    sequence_length = 10
    image_height = 299
    image_width = 299
    channels = 3
    
    return tf.random.uniform(
        (batch_size, sequence_length, image_height, image_width, channels),
        dtype=tf.float32
    )

