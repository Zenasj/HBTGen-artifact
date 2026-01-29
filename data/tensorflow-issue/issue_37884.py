# tf.random.uniform((1, 640, 360, 3), dtype=tf.float32) ← Inferred input shape from issue: batch=1, height=640, width=360, channels=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This is a simplified placeholder model mimicking SSD MobileNet V2 style detection output pattern,
    designed based on the typical postprocessed outputs given in the issue:
    - detection_boxes: [batch, num_boxes, 4]
    - detection_classes: [batch, num_boxes]
    - detection_scores: [batch, num_boxes]
    - num_detections: [batch]

    Since the original frozen graph is not fully re-creatable from the issue, this model generates
    dummy outputs with appropriate shapes and types to represent typical SSD outputs.
    """

    def __init__(self):
        super().__init__()
        # Assumed number of detections for output tensors (SSD MobileNet usually has 100 detections)
        self.num_detections_default = 100

        # Dummy layers to simulate outputs - all output tensors will have batch size 1 and the above shape
        self.dummy_boxes = tf.keras.layers.Lambda(
            lambda x: tf.zeros([tf.shape(x)[0], self.num_detections_default, 4], dtype=tf.float32),
            name="detection_boxes"
        )
        self.dummy_classes = tf.keras.layers.Lambda(
            lambda x: tf.zeros([tf.shape(x)[0], self.num_detections_default], dtype=tf.float32),
            name="detection_classes"
        )
        self.dummy_scores = tf.keras.layers.Lambda(
            lambda x: tf.zeros([tf.shape(x)[0], self.num_detections_default], dtype=tf.float32),
            name="detection_scores"
        )
        self.dummy_num_detections = tf.keras.layers.Lambda(
            lambda x: tf.ones([tf.shape(x)[0]], dtype=tf.float32),
            name="num_detections"
        )

    def call(self, inputs, training=False):
        """
        Forward pass generates dummy detection outputs.
        inputs: tensor of shape [batch, 640, 360, 3] as per the issue
        """
        boxes = self.dummy_boxes(inputs)
        classes = self.dummy_classes(inputs)
        scores = self.dummy_scores(inputs)
        num_dets = self.dummy_num_detections(inputs)
        # Return as a dictionary to mimic original output node names
        return {
            "detection_boxes": boxes,
            "detection_classes": classes,
            "detection_scores": scores,
            "num_detections": num_dets,
        }


def my_model_function():
    """
    Returns an instance of the MyModel class.
    No pretrained weights are loaded since original weights/frozen model not reproducible here.
    """
    return MyModel()


def GetInput():
    """
    Returns a random tensor shaped (1, 640, 360, 3) with dtype float32 matching the expected input shape.
    This models the input tensor `image_tensor` as specified in the issue and typical SSD MobileNet usage.
    """
    # Random float32 tensor in [0, 1) representing an image batch of size 1.
    return tf.random.uniform((1, 640, 360, 3), dtype=tf.float32)

