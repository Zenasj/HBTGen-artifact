# tf.random.uniform((B, 1), dtype=tf.float32) ← Assuming binary classification with scalar outputs per example

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    """
    This model serves as a placeholder to demonstrate usage of the custom weighted binary cross entropy loss.
    It outputs a single logit (or probability if from_logits=False) per example.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # Simple single dense layer with sigmoid activation to match weighted binary crossentropy expectations
        self.dense = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(inputs)


class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Custom weighted binary cross entropy loss supporting pos_weight and global weight scaling.

    pos_weight: scalar to scale the positive class loss contribution
    weight: scalar to scale the entire loss value
    from_logits: whether y_pred are logits or probabilities

    This class implements a call method replicating tf.nn.weighted_cross_entropy_with_logits
    behavior if from_logits=True, else manually computes the loss from probabilities.
    """

    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            with tf.name_scope('Weighted_Cross_Entropy'):
                # Manually calculated weighted cross entropy:
                # pos_weight scales positive class loss term
                # weight scales the entire loss scalar
                # epsilon (1e-6) added to prevent log(0)
                epsilon = 1e-6
                x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + epsilon)
                x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + epsilon)
                loss = (x_1 + x_2) * self.weight
                return loss
        # If from logits, use TF built-in weighted_cross_entropy_with_logits for numerical stability
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=self.pos_weight)
        return loss * self.weight


def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor compatible with MyModel.
    The model expects inputs of shape (batch_size, features).
    Since the model dense layer input size is inferred, assume input feature size of 10.
    
    Return shape: (batch_size=4, features=10)
    dtype: tf.float32
    """
    batch_size = 4
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

