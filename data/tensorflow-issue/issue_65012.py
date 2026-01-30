import tensorflow as tf
from tensorflow import keras

model = tf.keras.applications.DenseNet121 (
    include_top = True,
    weights = None,
    classes = 3,
    classifier_activation = "softmax"
)

mnist