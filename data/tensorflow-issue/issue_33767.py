# tf.random.uniform((B, img_w, img_h, 1), dtype=tf.float32) ← Assuming batch size B, grayscale image with given width and height

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Reshape, Dense, GRU, add, concatenate, Activation, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class FeatureExtraction(Layer):
    def __init__(self, conv_filters, pool_size, name='feature-extraction', **kwargs):
        super(FeatureExtraction, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same',
                            activation='relu', kernel_initializer='he_normal', name='conv1')
        self.conv2 = Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same',
                            activation='relu', kernel_initializer='he_normal', name='conv2')
        self.max1 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')
        self.max2 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        return self.max2(x)

    def get_config(self):
        return super(FeatureExtraction, self).get_config()


class FeatureReduction(Layer):
    def __init__(self, img_w, img_h, pool_size, conv_filters, name='feature-reduction', **kwargs):
        super(FeatureReduction, self).__init__(name=name, **kwargs)
        # After two max pooling layers each of pool_size, dimensions divided by pool_size^2
        # target shape is (img_w/(pool_size^2), (img_h/(pool_size^2)) * conv_filters)
        target_shape = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
        self.reshape = Reshape(target_shape=target_shape, name='reshape')
        self.dense = Dense(32, activation='relu', name='dense')

    def call(self, inputs):
        x = self.reshape(inputs)
        return self.dense(x)

    def get_config(self):
        return super(FeatureReduction, self).get_config()


class SequentialLearner(Layer):
    def __init__(self, name='sequential-learner', **kwargs):
        super(SequentialLearner, self).__init__(name=name, **kwargs)
        self.gru_1a = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru_1a')
        self.gru_1b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_1b')
        self.gru_2a = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru_2a')
        self.gru_2b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_2b')

    def call(self, inputs):
        x_1a = self.gru_1a(inputs)
        x_1b = self.gru_1b(inputs)
        x = add([x_1a, x_1b])
        x_2a = self.gru_2a(x)
        x_2b = self.gru_2b(x)
        return concatenate([x_2a, x_2b])

    def get_config(self):
        return super(SequentialLearner, self).get_config()


class Output(Layer):
    def __init__(self, output_size, name='output', **kwargs):
        super(Output, self).__init__(name=name, **kwargs)
        self.dense = Dense(output_size, kernel_initializer='he_normal', name='dense')
        self.softmax = Activation('softmax', name='softmax')

    def call(self, inputs):
        x = self.dense(inputs)
        return self.softmax(x)

    def get_config(self):
        return super(Output, self).get_config()


class MyModel(tf.keras.Model):
    def __init__(self, output_size, img_w, img_h, max_text_len, name='OCRNet', **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)

        # Parameters from original model
        self.conv_filters = 16
        self.pool_size = 2
        self.output_size = output_size
        self.img_w = img_w
        self.img_h = img_h
        self.max_text_len = max_text_len

        # Submodules
        self.feature_extraction = FeatureExtraction(conv_filters=self.conv_filters, pool_size=self.pool_size)
        self.feature_reduction = FeatureReduction(img_w=img_w, img_h=img_h, pool_size=self.pool_size,
                                                  conv_filters=self.conv_filters)
        self.sequential_learner = SequentialLearner()
        self.output_layer = Output(output_size)

    def call(self, inputs):
        # inputs is a single tensor: image inputs
        # This is a re-implemented forward pass with only image input since
        # original model had multiple inputs for CTC loss implementation.
        # For direct call, only image input processed.

        x = self.feature_extraction(inputs)
        x = self.feature_reduction(x)
        x = self.sequential_learner(x)
        predictions = self.output_layer(x)
        return predictions

    def ctc_loss(self, y_true, y_pred, input_length, label_length):
        # y_pred shape: (batch, time, classes)
        # Trim first 2 timesteps as original model does
        y_pred_proc = y_pred[:, 2:, :]
        return K.ctc_batch_cost(y_true, y_pred_proc, input_length, label_length)


def my_model_function():
    # Return an instance of MyModel with example parameters
    output_size = 80  # Example: number of character classes + blank for CTC
    img_w = 128       # example image width
    img_h = 32        # example image height (height as different dimension from width)
    max_text_len = 32 # maximum length of text sequence
    model = MyModel(output_size=output_size, img_w=img_w, img_h=img_h, max_text_len=max_text_len)
    return model


def GetInput():
    # Generates a batch of example input images matching expected shape
    # Assume batch size 4 for example
    batch_size = 4
    # Tensor shape NHWC (channels_last)
    # Single channel grayscale
    # dtype float32 consistent with model
    img_w = 128
    img_h = 32
    input_tensor = tf.random.uniform(shape=(batch_size, img_w, img_h, 1), dtype=tf.float32)
    return input_tensor

