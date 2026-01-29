# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Example inferred input shape from provided minimal repro: (1, 10, 10, 3)
import tensorflow as tf

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mIoU', dtype=None):
        # Call super with required num_classes parameter and optional name, dtype
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Override update_state to convert logits/probabilities y_pred to class labels with argmax
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        # Simple conv2d layer producing per-pixel class logits/probabilities for segmentation
        self.conv = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same')
        self.softmax = tf.keras.layers.Activation('softmax')
        # Using updated MeanIoU metric to correctly handle sparse labels and logits
        self.mean_iou_metric = UpdatedMeanIoU(num_classes=num_classes)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.softmax(x)
        return x

    def compile(self, optimizer, loss):
        # Compile model with given optimizer and loss function, and include MeanIoU metric
        super(MyModel, self).compile(optimizer=optimizer, loss=loss, metrics=[self.mean_iou_metric])

def my_model_function():
    # Return an instance of MyModel with 10 classes - matching typical example in the issue comments
    model = MyModel(num_classes=10)

    # Compile model using sparse categorical crossentropy and SGD optimizer as in the repro example
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss=loss_obj)

    return model

def GetInput():
    # Generate a batch of 1 example with shape (10, 10, 3) [height=10, width=10, channels=3],
    # dtype float32 matching typical RGB image input.
    # This matches the minimal repro example from the issue.
    B, H, W, C = 1, 10, 10, 3
    input_tensor = tf.random.uniform((B, H, W, C), minval=-1, maxval=1, dtype=tf.float32)
    return input_tensor

