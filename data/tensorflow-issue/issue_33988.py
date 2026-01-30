import tensorflow as tf
from tensorflow import keras

class SparseCategoricalCrossentropyIgnoreLabel(tf.keras.losses.Loss):
    """Computes the crossentropy loss between the labels and predictions,
        with ignored label.
        Derived directly from tf.keras.losses.Loss
    """
    def __init__(self, ignore_label=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='sparse_categorical_crossentropy_ignore_label'):
        super().__init__(reduction=reduction, name=name)
        self.ignore_label = ignore_label
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Keep only batch dimension for the final loss
        """
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]))
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))
        loss_tensor = tf.keras.backend.sparse_categorical_crossentropy(
                y_true, y_pred,
                from_logits=self.from_logits)
        mask = tf.ones_like(loss_tensor)
        if self.ignore_label is not None:
            mask = (y_true != self.ignore_label)
            mask = tf.squeeze(mask, [-1])
            loss_tensor = tf.where(mask, loss_tensor, 0)
        
        num_elements = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
        loss_tensor = tf.reduce_sum(loss_tensor, axis=1)/num_elements
        loss_tensor = tf.where(num_elements > 0, loss_tensor, 0)
        return loss_tensor

class SparseCategoricalCrossentropyIgnoreLabel(tf.keras.losses.SparseCategoricalCrossentropy):
    """Computes the crossentropy loss between the labels and predictions,
        with ignored label.
        Derived from tf.keras.losses.SparseCategoricalCrossentropy
    """
    def __init__(self, ignore_label=None, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='sparse_categorical_crossentropy_ignore_label'):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)
        self.ignore_label = ignore_label
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        if self.ignore_label is not None:
            if sample_weight is not None:
                sample_weight = tf.where(y_true == self.ignore_label, 0, sample_weight)
            else:
                sample_weight = y_true != self.ignore_label
        return super().__call__(y_true, y_pred, sample_weight)