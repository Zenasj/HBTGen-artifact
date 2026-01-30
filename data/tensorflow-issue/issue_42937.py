from tensorflow import keras

import sys
sys.path.append("hmr2.0/src/")

import tensorflow as tf

from main.model import Model
from notebooks.vis_util import preprocess_image
from main.config import Config


def main():
    model = Model()
    config = Config()
    generator = model.generator   
    
    original_img, input_img, params = preprocess_image('hmr2.0/src/notebooks/images/coco4.png', \
                                                        config.ENCODER_INPUT_SHAPE[0])
    
    if len(tf.shape(input_img)) is not 4:
        # [224, 224, 3] -> [1, 224, 224, 3]
        input_img = tf.expand_dims(input_img, 0) 
    generator._set_inputs(input_img)

    converter = tf.lite.TFLiteConverter.from_keras_model(generator)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()


if __name__ == '__main__':
    main()

class Regressor(tf.keras.Model):
    def __init__(self):
        super(Regressor, self).__init__(name='regressor')
        self.config = Config()

        self.mean_theta = tf.Variable(model_util.load_mean_theta(), name='mean_theta', trainable=True)

        self.fc_one = layers.Dense(1024, name='fc_0')
        self.dropout_one = layers.Dropout(0.5)
        self.fc_two = layers.Dense(1024, name='fc_1')
        self.dropout_two = layers.Dropout(0.5)
        variance_scaling = tf.initializers.VarianceScaling(.01, mode='fan_avg', distribution='uniform')
        self.fc_out = layers.Dense(85, kernel_initializer=variance_scaling, name='fc_out')

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, 2048)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        batch_theta = tf.tile(self.mean_theta, [batch_size, 1])
        thetas = tf.TensorArray(tf.float32, self.config.ITERATIONS)
        for i in range(self.config.ITERATIONS):
            # [batch x 2133] <- [batch x 2048] + [batch x 85]
            total_inputs = tf.concat([inputs, batch_theta], axis=1)
            batch_theta = batch_theta + self._fc_blocks(total_inputs, **kwargs)
            thetas = thetas.write(i, batch_theta)

        return thetas.stack()

    def _fc_blocks(self, inputs, **kwargs):
        x = self.fc_one(inputs, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_one(x, **kwargs)
        x = self.fc_two(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_two(x, **kwargs)
        x = self.fc_out(x, **kwargs)
        return x