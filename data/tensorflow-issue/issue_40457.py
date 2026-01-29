# tf.random.uniform((None, 16), dtype=tf.int32)  # inferred from issue input shape (None, 16)

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=25):
        super().__init__()
        # A simple example model that outputs a tensor matching the reported output shape
        # Input shape: (None, 16)
        # Output shape: (None, 25)
        # We use a simple Dense layer here as the issue's model is not fully described.
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass
        return self.dense(inputs)

def myLoss(originalLossFunc, weightsList):
    """
    Custom loss function to apply per-class weights during training.

    - originalLossFunc: A callable loss function, e.g. tf.keras.losses.CategoricalCrossentropy()
    - weightsList: List or 1D tensor of shape (num_classes,) containing weights for each class

    The loss multiplies the per-sample loss by the class weight corresponding to the true class.
    """

    def lossFunc(y_true, y_pred):
        # Assuming y_true is one-hot encoded with shape (batch, num_classes)

        # Get the class index for each sample
        axis = -1  # channel last
        classSelectors = tf.argmax(y_true, axis=axis, output_type=tf.int32)

        # Create boolean mask for each class
        weightsList_tensor = tf.constant(weightsList, dtype=tf.float32)  # shape: (num_classes,)

        # Gather weights based on the true class indices
        weightMultiplier = tf.gather(weightsList_tensor, classSelectors)  # shape: (batch,)

        # Compute the original loss per sample (lossFunc expects shape (batch, num_classes))
        # originalLossFunc typically returns mean loss over batch by default, but here it's used per-sample.
        # To get per-sample loss, use reduction=NONE option.
        # So ensure originalLossFunc has reduction=tf.keras.losses.Reduction.NONE
        loss = originalLossFunc(y_true, y_pred)  # shape: (batch,)
        # Multiply loss by weights
        weighted_loss = loss * weightMultiplier
        return tf.reduce_mean(weighted_loss)

    return lossFunc

def my_model_function():
    """
    Create and return an instance of MyModel.
    """
    model = MyModel()
    return model

def GetInput():
    """
    Return a random tensor input compatible with MyModel.

    Input shape inferred from issue to be (batch, 16), dtype int32.
    Batch size is arbitrarily chosen (e.g., 8).
    """
    B = 8
    H = 16  # Input length
    # Generate random int32 tensor matching input shape
    x = tf.random.uniform((B, H), minval=0, maxval=100, dtype=tf.int32)
    return x

