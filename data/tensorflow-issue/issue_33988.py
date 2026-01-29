# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape with batch and spatial dims (common for image tensors)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple model architecture for demonstration:
        # Assume classification with logits output for a 10-class problem
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(10)  # 10 output classes
        
        # Instantiate two loss functions reflecting the discussion:
        
        # 1. Loss derived directly from tf.keras.losses.Loss,
        # works correctly according to issue comments
        self.loss_direct = SparseCategoricalCrossentropyIgnoreLabel_Direct(ignore_label=-1, from_logits=True)
        
        # 2. Loss derived from tf.keras.losses.SparseCategoricalCrossentropy,
        # reported to cause incorrect behavior in .fit()
        self.loss_from_sparse = SparseCategoricalCrossentropyIgnoreLabel_FromSparse(ignore_label=-1, from_logits=True)
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        logits = self.logits(x)
        return logits
    
    def compare_losses(self, y_true, y_pred):
        # Compute loss values from both custom losses
        loss_direct_val = tf.reduce_mean(self.loss_direct(y_true, y_pred))
        loss_sparse_val = tf.reduce_mean(self.loss_from_sparse(y_true, y_pred))
        
        # Compare loss values and output difference
        loss_diff = loss_direct_val - loss_sparse_val
        
        # Also output boolean if they are close (within a small tolerance)
        close = tf.abs(loss_diff) < 1e-6
        
        return {
            'loss_direct': loss_direct_val,
            'loss_from_sparse': loss_sparse_val,
            'difference': loss_diff,
            'close': close
        }

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Generate a random input tensor simulating a batch of RGB images of size 32x32
    batch_size = 8
    height = 32
    width = 32
    channels = 3
    x = tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
    
    # Generate random integer labels in [0,9], with occasional ignore_label set to -1
    # Shape: (batch_size,) corresponding to sparse categorical labels
    y = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)
    # Randomly set some labels to ignore_label = -1
    ignore_mask = tf.random.uniform((batch_size,)) < 0.2  # ~20% ignored labels
    y = tf.where(ignore_mask, -1, y)
    return x, y


# Implementation of two custom loss functions as per issue discussion:

class SparseCategoricalCrossentropyIgnoreLabel_Direct(tf.keras.losses.Loss):
    """
    Computes sparse categorical crossentropy loss with an ignored label, derived directly from tf.keras.losses.Loss.
    This version was noted to produce correct results in Keras .fit().
    """
    def __init__(self, ignore_label=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='sparse_categorical_crossentropy_ignore_label_direct'):
        super().__init__(reduction=reduction, name=name)
        self.ignore_label = ignore_label
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        """
        y_true: shape (batch, ...) integer labels (sparse)
        y_pred: shape (batch, ..., num_classes)
        """
        y_true = tf.cast(y_true, tf.int32)
        # Flatten spatial dims except for batch, if present
        y_true_reshaped = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
        y_pred_reshaped = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))
        
        # Compute loss per element
        loss_tensor = tf.keras.backend.sparse_categorical_crossentropy(
            y_true_reshaped, y_pred_reshaped, from_logits=self.from_logits)
        
        # Build mask: 1 for valid labels, 0 for ignored labels
        mask = tf.ones_like(loss_tensor)
        if self.ignore_label is not None:
            mask = tf.cast(tf.not_equal(y_true_reshaped, self.ignore_label), tf.float32)
        
        # Multiply loss by mask to ignore certain labels
        loss_tensor = loss_tensor * mask
        
        # Normalize: sum loss / number of valid elements per batch item
        num_elements = tf.reduce_sum(mask, axis=1)
        loss_per_sample = tf.math.divide_no_nan(tf.reduce_sum(loss_tensor, axis=1), num_elements)
        
        # Loss is batch average by default reduction AUTO
        return loss_per_sample

class SparseCategoricalCrossentropyIgnoreLabel_FromSparse(tf.keras.losses.SparseCategoricalCrossentropy):
    """
    Computes sparse categorical crossentropy loss with an ignored label,
    derived by subclassing tf.keras.losses.SparseCategoricalCrossentropy.
    This version was reported to cause inconsistent training loss results in issue.
    """
    def __init__(self, ignore_label=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='sparse_categorical_crossentropy_ignore_label_from_sparse'):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)
        self.ignore_label = ignore_label
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        if self.ignore_label is not None:
            if sample_weight is not None:
                sample_weight = tf.where(tf.equal(y_true, self.ignore_label), 0., sample_weight)
            else:
                sample_weight = tf.cast(tf.not_equal(y_true, self.ignore_label), tf.float32)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

