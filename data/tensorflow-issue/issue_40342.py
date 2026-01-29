# tf.random.uniform((GLOBAL_BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST data with batch size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicate the minimal Sequential model from the issue as functional layers inside the subclassed model
        self.conv = tf.keras.layers.Conv2D(filters=32, strides=1, kernel_size=(4,4), input_shape=(28,28,1))
        self.activation = tf.keras.layers.Activation('relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.batchnorm(x, training=training)  # pass training flag for BatchNorm layer
        x = self.flatten(x)
        x = self.dense(x)
        return x


class SparseCategoricalLoss(tf.keras.losses.Loss):
    """
    Custom loss class replicating the SparseCategoricalLoss described in the issue. 
    This custom loss slices inputs to num_classes for y_true and y_pred and applies sparse categorical crossentropy.
    Includes 'loss_weight' to scale loss, and supports from_logits flag.
    """
    def __init__(self, num_classes, name='SparseCategoricalLoss', from_logits=False, loss_weight=1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.loss_weight = loss_weight

        # Create internal SparseCategoricalCrossentropy with reduction NONE for custom weighting
        self._sce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction=tf.keras.losses.Reduction.NONE,
            name=name,
        )
        
    def loss_fn(self, y_true, y_pred):
        # According to original code, slicing first num_classes elements on axis=1 of y_true and y_pred
        label = y_true[:, 0:self.num_classes]
        logit = y_pred[:, 0:self.num_classes]
        loss = self._sce(label, logit)
        loss *= self.loss_weight
        return loss
    
    def call(self, y_true, y_pred):
        total_loss = self.loss_fn(y_true, y_pred)
        return total_loss
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'from_logits': self.from_logits,
            'loss_weight': self.loss_weight,
        })
        return config


def my_model_function():
    """
    Returns compiled MyModel instance as per original setup using MirroredStrategy scope.
    Uses RMSprop optimizer with specified params, compiles model with custom SparseCategoricalLoss 
    and accuracy metric.
    """
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = MyModel()
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            epsilon=1.0,
            momentum=0.9,
            rho=0.9
        )
        loss = SparseCategoricalLoss(num_classes=10, from_logits=True, name='categorical_loss')
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    return model


def GetInput():
    """
    Returns a random tensor representative of the expected input to MyModel.
    Shape: [GLOBAL_BATCH_SIZE, 28, 28, 1]
    Using float32 values uniformly between 0 and 1.
    The batch size is inferred from the example: batch_size_per_gpu * number_of_gpus.
    For standalone reproducibility, we choose batch size 16 * 1 gpu = 16 by default.
    """
    # Assumption: Single GPU scenario. If distributed, batch size should be accordingly scaled.
    batch_size = 16  # Using batch size per GPU as in the issue example
    input_shape = (batch_size, 28, 28, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)


# Additional notes:
# - This model matches the minimal MNIST example from the issue.
# - The loss class faithfully replicates the original custom SparseCategoricalLoss.
# - The model is designed to be compiled with a distribution strategy externally.
# - The input shape and dtype matches the usage inside the provided code snippets.
# - The SparseCategoricalLoss slices input tensors on axis=1 which assumes y_true and y_pred have at least 'num_classes' width in axis 1.
#   For MNIST typical labels, y_true is usually (batch,) of class indices. The original code suggests y_true shape having more dimensions or custom format,
#   but here we follow the original slicing as is for faithfulness.

