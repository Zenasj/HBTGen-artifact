# tf.random.uniform((1, 224, 224, 2), dtype=tf.float32) ← Assuming batch size 1, 224x224 image, 2 classes (logits)

import tensorflow as tf

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)
    
    @tf.function
    def __call__(self, y_true, y_pred, sample_weight=None):
        # y_pred shape: (batch, height, width, num_classes)
        # y_true shape: (batch, height, width) sparse labels
        
        # Convert logits to predicted classes by argmax on the last dim
        if y_pred.shape.ndims is not None and y_pred.shape.ndims > 1:
            y_pred = tf.math.argmax(y_pred, axis=-1)
        
        # Flatten y_true and y_pred to 1D for confusion matrix
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        return super().__call__(y_true, y_pred, sample_weight)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple example backbone for semantic segmentation logits output
        # This model produces output shape (batch, 224, 224, 2)
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(2, 1)  # 2 classes logits
        
        # Instantiate UpdatedMeanIoU metric internally for demonstration
        self.mean_iou_metric = UpdatedMeanIoU(num_classes=2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        logits = self.conv3(x)  # shape (B, 224, 224, 2)
        return logits
    
    def compute_mean_iou(self, y_true, y_pred):
        # y_true shape: (B, 224, 224) sparse int labels
        # y_pred shape: (B, 224, 224, 2) logits
        return self.mean_iou_metric(y_true, y_pred)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (1, 224, 224, 2)
    # This input shape matches the expected input shape for MyModel
    # assuming input images have 2 channels (e.g. could be 2-class semantic segmentation feature maps)
    # This is an assumption since input shape was not fully specified.
    # Usually input images have 3 channels (RGB), but this example uses 2 as per error context.
    return tf.random.uniform((1, 224, 224, 2), dtype=tf.float32)

