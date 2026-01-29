# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is inferred as any 4D tensor for convolutional model, e.g. (batch, height, width, channels)

import tensorflow as tf
from tensorflow.keras import layers, losses, backend as K

# Define a placeholder focal_loss function since the original is external.
# In practice, replace this with the user-provided focal_loss implementation.
def focal_loss(y_true, y_pred, mask=None):
    # Simplified dummy version assuming y_true/y_pred shape is (batch, n, classes)
    # The mask parameter is used for weighting; if mask is provided, apply it.
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    if mask is not None:
        # mask is assumed broadcastable to loss shape
        loss = loss * mask
    return loss

class FocalLoss(losses.Loss):
    def __init__(self,
                 num_class,
                 num_sub_catg_class,
                 global_batch_size,
                 input_shape,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 lambda_conf=100.0,
                 lambda_offsets=1.0,
                 class_weights=1.0,
                 name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.global_batch_size = global_batch_size
        self.num_class = num_class
        self.num_sub_catg_class = num_sub_catg_class
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.class_weights = class_weights
        self.num_rows, self.num_cols = input_shape[:2]

    def call(self, y_true, y_pred, sample_weight=None):
        # sample_weight is unused because `call` signature must be (y_true, y_pred)
        # If weighting needs to be passed, user must wrap or handle externally.
        
        num_rows, num_cols = self.num_rows, self.num_cols
        class_mask_tensor = 1.0
        class_sub_catg_mask_tensor = 1.0

        # The sample_weight is not standard part of call signature; can be handled outside
        # or integrated if customizing further.
        
        class_y_true = y_true[:, :, :, :self.num_class]
        sub_catg_class_y_true = y_true[:, :, :, self.num_class:self.num_class+self.num_sub_catg_class]

        class_y_pred = y_pred[:, :, :, :self.num_class]
        sub_catg_class_y_pred = y_pred[:, :, :, self.num_class:self.num_class+self.num_sub_catg_class]

        class_y_true = tf.reshape(class_y_true, [self.global_batch_size, num_rows * num_cols, self.num_class])
        class_y_pred = tf.reshape(class_y_pred, [self.global_batch_size, num_rows * num_cols, self.num_class])
        class_loss = focal_loss(class_y_true, class_y_pred, mask=class_mask_tensor)

        sub_catg_class_y_true = tf.reshape(sub_catg_class_y_true, [self.global_batch_size, num_rows * num_cols, self.num_sub_catg_class])
        sub_catg_class_y_pred = tf.reshape(sub_catg_class_y_pred, [self.global_batch_size, num_rows * num_cols, self.num_sub_catg_class])
        sub_catg_class_loss = focal_loss(sub_catg_class_y_true, sub_catg_class_y_pred, mask=class_sub_catg_mask_tensor)

        mean_class_loss = tf.nn.compute_average_loss(tf.squeeze(class_loss), global_batch_size=self.global_batch_size)
        mean_sub_catg_class_loss = tf.nn.compute_average_loss(tf.squeeze(sub_catg_class_loss), global_batch_size=self.global_batch_size)

        total_loss = mean_class_loss + mean_sub_catg_class_loss
        return total_loss


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reproduce the simple Conv2D classification model from issue example
        self.conv1 = layers.Conv2D(6, (3, 3), activation='relu')
        self.pool1 = layers.AveragePooling2D()
        self.conv2 = layers.Conv2D(16, (3, 3), activation='relu')
        self.pool2 = layers.AveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(10, activation='softmax')

        # Also instantiate a focal loss submodule with dummy params for demonstration
        # This is illustrative of fusion: model + loss in one class
        # We assume input of shape (batch, 28, 28, 1) as MNIST usually
        self.focal_loss = FocalLoss(num_class=10, num_sub_catg_class=0, global_batch_size=1, input_shape=(28, 28, 1))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

    def compute_focal_loss(self, y_true, y_pred):
        # Expose a method to compute fused custom focal loss if needed, 
        # supports sample_weight as None for standard usage
        return self.focal_loss(y_true, y_pred)


def my_model_function():
    # Return an instance of MyModel, weights uninitialized/random
    return MyModel()


def GetInput():
    # MNIST-style input: batch=1, height=28, width=28, channels=1
    # dtype float32 normalized to [-1,1] as per input preprocessing in issue.
    input_shape = (1, 28, 28, 1)
    x = tf.random.uniform(input_shape, minval=-1.0, maxval=1.0, dtype=tf.float32)
    return x

