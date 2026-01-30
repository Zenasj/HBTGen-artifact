from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Outer(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, train_bn=False):
        return self.bn(x, training=train_bn)

class Infer(tf.Module):
    def __init__(self):
        super().__init__()

        # Decorate the inference function with tf.function
        self.infer_ = tf.function(self.infer, input_signature=[
             tf.TensorSpec([1, 64, 64, 8], tf.float32, 'prev_img')])

        self.outer = Outer()

    def infer(self, input):
        return self.outer(input, train_bn=False)

# Create model
infer = Infer()

# Save the trained model
signature_dict = {'infer': infer.infer_}
saved_model_dir = '/tmp/saved_model'
tf.saved_model.save(infer, saved_model_dir, signature_dict)