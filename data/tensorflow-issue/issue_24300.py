# tf.random.uniform((1, 640, 640, 3), dtype=tf.float32) ← inferred input shape from feed description in graph.config.pbtxt

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Placeholder for SSD-ResNet50 FPN submodule
        # The original model is complex and pretrained; here we simulate with a few conv layers
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])
        # Placeholder box predictor simulating detection outputs
        self.box_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(100 * 4, activation=None),  # 100 boxes, 4 coords each
            tf.keras.layers.Reshape((100, 4)),
        ])
        # Placeholder score predictor
        self.score_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='sigmoid'),  # 100 scores
        ])
        # Placeholder class predictor
        self.class_predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='softmax'),  # 100 class probabilities (simplified)
        ])

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs shape: (1, 640, 640, 3)
        features = self.backbone(inputs)  # shape: (1, 128,)
        boxes = self.box_predictor(features)  # (1, 100, 4)
        scores = self.score_predictor(features)  # (1, 100)
        classes = self.class_predictor(features)  # (1, 100)

        # Simulate postprocessing outputs matching detection outputs in the config fetch:
        # detection_boxes: float32 [1, num_detections, 4]
        # detection_scores: float32 [1, num_detections]
        # detection_classes: float32/int [1, num_detections]
        # num_detections: float32/int [1]

        num_detections = tf.constant([100], dtype=tf.int32)  # fixed for simplicity

        # Cast classes to int32 class ids (taking argmax simplified here)
        detection_classes = tf.argmax(classes, axis=-1, output_type=tf.int32)  # (1, 100)

        return {
            'detection_boxes': boxes,
            'detection_scores': scores,
            'detection_classes': tf.cast(detection_classes, tf.float32),
            'num_detections': tf.cast(num_detections, tf.float32)
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching input shape from the feed section:
    # shape (1, 640, 640, 3), dtype float32 (image_tensor)
    return tf.random.uniform((1, 640, 640, 3), dtype=tf.float32)

