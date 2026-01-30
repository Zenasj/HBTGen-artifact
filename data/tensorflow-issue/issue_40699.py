from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_model_optimization as tfmot

class MyNet(tf.keras.Model):

    def __init__(self, filter_nbr=8):
        """Init MyNet fields."""
        super(MyNet, self).__init__()

        self.conv11 = tf.keras.layers.Conv2D(filter_nbr, kernel_size=3, padding='SAME')

    def call(self, inputs):
        """Forward method."""
        # Stage 1
        x11 = (tf.nn.relu(self.bn11(self.conv11(inputs))))

        return x11

model = MyNet()
model = tfmot.quantization.keras.quantize_model(model)

# Got:
# ValueError: `to_quantize` can only either be a tf.keras Sequential or Functional model.

# expected no error