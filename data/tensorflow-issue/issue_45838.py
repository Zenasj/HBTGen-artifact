# tf.random.uniform((B, None, None, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras import models, optimizers

# Assumptions:
# - Input shape is (batch_size, height, width, 3), height and width are dynamic (None).
# - The model structure (RPN + classifier) is not fully detailed, so this is a dummy skeleton embedding
#   the classification loss regression function shown in the issue.
# - The key demonstrated loss function is implemented here as a callable loss layer inside MyModel.
# - GetInput returns a random input tensor matching a typical image input shape for testing.
# - The model forward returns a dummy output tensor matching what classifier regression might produce.

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        
        # We don't have the full model definitions (e.g. share_layer, RPN, classifier),
        # so create placeholder layers to simulate output shapes and test code compatibility.
        
        # Placeholder conv layer for feature extraction (ResNet50 share layer would output feature maps)
        self.feature_extractor = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(None, None, 3),
            pooling=None,
            classes=num_classes,
            weights=None
        )
        
        # Dummy classifier regression head outputting bounding box coords:
        # Output shape = (batch_size, num_rois, num_classes * 4)
        # Here we simulate num_rois as 32 (arbitrary) for testing purpose.
        self.num_rois = 32
        
        # Use a simple Dense layer to map extracted features to the regression coords
        # Since feature_extractor output spatial dims are unknown at runtime, we flatten spatially then Dense.
        self.flatten = tf.keras.layers.Flatten()
        self.dense_regress = tf.keras.layers.Dense(self.num_rois * self.num_classes * 4)
        
    def call(self, inputs, training=False):
        # inputs is expected as a list or tuple: [images, rois]
        # images shape: (batch_size, H, W, 3)
        # rois shape: (batch_size, num_rois, 4)
        # For demonstration, we use only images as input
        
        images = inputs if not isinstance(inputs, (list, tuple)) else inputs[0]
        
        # Extract features (shape will be (batch_size, H_out, W_out, 2048))
        features = self.feature_extractor(images, training=training)
        
        # Flatten features and map to regression output
        x = self.flatten(features)
        regressions = self.dense_regress(x)
        
        # Reshape to (batch_size, num_rois, num_classes * 4)
        regressions = tf.reshape(regressions, (-1, self.num_rois, self.num_classes * 4))
        
        return regressions
    
    def class_loss_regr(self, y_true, y_pred):
        """
        Custom smooth L1 loss (also called Huber loss variant) for bounding box regression,
        adapted for Faster RCNN multi-class regression output.
        
        y_true shape: (batch_size, num_rois, num_classes * 8)
            - [:, :, :4*num_classes] is label index mask (1 or 0) for which class applies per ROI
            - [:, :, 4*num_classes:] is ground truth bounding box coordinates, corresponding to y_pred
        y_pred shape: (batch_size, num_rois, num_classes * 4)
        
        Returns scalar loss.
        """
        epsilon = 1e-4
        batch_size = tf.shape(y_true)[0]
        
        # Loop over batch is unavoidable here due to masking logic per sample
        regr_loss = 0.0
        
        for i in tf.range(batch_size):
            y_true_sample = y_true[i]
            y_pred_sample = y_pred[i]
            
            # Compute difference between ground truth boxes and predicted boxes
            x = y_true_sample[:, 4 * self.num_classes:] - y_pred_sample
            
            x_abs = tf.abs(x)
            x_bool = tf.cast(x_abs <= 1.0, 'float32')
            
            # Apply smooth L1 loss formula, weighted by the masks present in y_true for class indices
            loss_sample = 4 * tf.reduce_sum(
                y_true_sample[:, :4 * self.num_classes] *
                (x_bool * 0.5 * x * x + (1 - x_bool) * (x_abs - 0.5))
            ) / (tf.reduce_sum(epsilon + y_true_sample[:, :4 * self.num_classes]))
            
            regr_loss += loss_sample
        
        return regr_loss / tf.cast(batch_size, tf.float32)


def my_model_function():
    # Instantiate MyModel with default 20 classes (VOC dataset)
    model = MyModel(num_classes=20)
    return model


def GetInput():
    # Provide a random input tensor compatible with MyModel:
    # Input is a batch of RGB images with dynamic size:
    # For testing, generate batch size 2, height 224, width 224, channels 3
    batch_size = 2
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

