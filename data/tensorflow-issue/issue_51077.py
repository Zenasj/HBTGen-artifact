# tf.random.uniform((B, 13), dtype=tf.float32)  ← Arbitrary batch size B, 13 features as in heart.csv example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a Sequential model replicating the example in the issue
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(13,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

# Fixes and notes:
# - The error in the issue arises because tf.math.confusion_matrix expects 1D tensors of int labels,
#   but in the original metric code y_true and y_pred have shape (batch_size,1).
# - They need to be squeezed to 1D.
# - The confusion matrix is for discrete labels, but model outputs are probabilities; need thresholding.
# - Also, the indexing of confusion matrix in the original code is wrong (confusion_matrix shape is [num_classes,num_classes]):
#   matrix[0][0] = True Negative (assuming 0 = negative class)
#   matrix[1][1] = True Positive
#   matrix[0][1] = False Positive
#   matrix[1][0] = False Negative
# The user's code was mixing those indices.
# This code below shows how to get confusion matrix correctly for binary classification.

class MyScore(tf.keras.metrics.Metric):
    def __init__(self, name='my_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.tn = self.add_weight(name='tn', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)
        self.score = self.add_weight(name='score', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true and y_pred come with shape (batch_size, 1), convert to 1D.
        y_true = tf.reshape(y_true, [-1])
        # Convert probabilities to 0/1 class predictions
        y_pred = tf.reshape(y_pred, [-1])
        y_pred_label = tf.cast(y_pred > 0.5, tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)

        # Compute confusion matrix for binary classes 0 and 1
        matrix = tf.math.confusion_matrix(y_true_int, y_pred_label, num_classes=2, dtype=tf.float32)

        # Note:
        # confusion_matrix layout:
        # [[TN FP]
        #  [FN TP]]
        tn = matrix[0, 0]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        tp = matrix[1, 1]

        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

        # Compute cost and score as in original logic, but now carefully:
        # cost = (tn + fp) / (tp + fn) might divide by zero, protect with small epsilon
        denom = tp + fn
        denom = tf.where(denom == 0, tf.constant(1.0), denom)

        cost = (tn + fp) / denom
        misprediction_cost = fn * cost + fp
        weighted_samples = tn + fp + cost * (tp + fn)
        score = 1.0 - (misprediction_cost / weighted_samples)
        self.score.assign(score)

    def result(self):
        return self.score

    def reset_state(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
        self.score.assign(0)

def my_model_function():
    # Return an instance of MyModel.
    return MyModel()

def GetInput():
    # Generate a random tensor with batch size 512 and 13 features,
    # matching the example heart.csv features input shape.
    batch_size = 512
    return tf.random.uniform((batch_size, 13), dtype=tf.float32)

