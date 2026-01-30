import math
import tensorflow as tf

class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    pos_weight: Scalars the effec on loss by the positive class by whatever is passed into it.
    weight: Scalars all the loss. Can be used to increase scalar of negative weight only by passing 1/weight to pos_weight. 
            To affect pos_weight even more after this multiply in the other scalar you had in mind for it
    """
    def __init__(self, pos_weight, weight, from_logits=False, reduction=keras.losses.Reduction.AUTO, name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            with tf.name_scope('Weighted_Cross_Entropy'):
                # Manually calculated the weighted cross entropy. Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)) where z are labels, x is logits, and q is the weight.
                # Since the values passed are from sigmoid (assumably in this case) sigmoid(x) will be replaces with y_pred
                x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6) # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
                x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6) # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
                return tf.add(x_1, x_2) * self.weight # Must be negative as it is maximized when passed to optimizers
        # Use built in function
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight