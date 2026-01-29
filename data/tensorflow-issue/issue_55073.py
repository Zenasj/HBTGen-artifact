# tf.random.uniform((1, 300, 300, 3), dtype=tf.uint8) ← Inferred input shape and dtype based on example and representative dataset

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the TFHub SSD MobileNet V2 model layer
        # Note: The original hub.KerasLayer outputs detection boxes, scores, classes, etc.
        # Using from TFHub URL 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2' as in the issue.
        self.detector = hub.KerasLayer('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

    def call(self, inputs, training=False):
        # The input is expected to be uint8 image tensor
        # The TFHub detector expects input normalized internally, so just forward inputs
        outputs = self.detector(inputs)
        # Outputs dictionary typically includes:
        # 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'.
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide a random sample input tensor that matches expected model input:
    # Batch size 1, 300x300 image, 3 channels (RGB), uint8 type
    # Using tf.random.uniform with dtype uint8 and shape (1,300,300,3)
    input_tensor = tf.random.uniform(shape=(1, 300, 300, 3), minval=0, maxval=256, dtype=tf.uint8)
    return input_tensor

