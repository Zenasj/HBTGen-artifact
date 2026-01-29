# tf.random.uniform((BATCH_SIZE, 192, 192, 3), dtype=tf.float32)
import tensorflow as tf

# This model encapsulates the MobileNetV2 feature extractor with a classifier head.
# It supports an explicit `training` argument in the call to ensure batchnorm layers behave correctly
# during custom training loops, addressing the original issue where omission of training=True caused no learning.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        IMAGE_SIZE = [192, 192]
        self.pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',
            # training arg not accepted here, must be passed during forward call
        )
        self.pretrained_model.trainable = True
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(5, activation='softmax')
    
    def call(self, inputs, training=False):
        # Pass training flag to pretrained_model to ensure BatchNorm layers work correctly
        x = self.pretrained_model(inputs, training=training)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (BATCH_SIZE, 192, 192, 3)
    # Use float32 values in [0,1] range as images normally are normalized so.
    BATCH_SIZE = 64  # consistent with original code's batch size
    IMAGE_SIZE = [192, 192]
    return tf.random.uniform(shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), minval=0, maxval=1, dtype=tf.float32)

