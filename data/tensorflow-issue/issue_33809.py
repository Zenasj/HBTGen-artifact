# tf.random.uniform((batch_size, img_w, img_h, 1), dtype=tf.float32) ← Assuming batch size dynamic, img_w and img_h as per OCRNet initialization

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
        # Target shape reshapes to (width after pooling^2, height after pooling^2 * conv_filters)
        target_shape = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
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

class OCRNet(Model):
    def __init__(self, output_size, img_w, img_h, max_text_len, name='OCRNet', **kwargs):
        conv_filters = 16
        pool_size = 2

        feature_extraction = FeatureExtraction(conv_filters=conv_filters, pool_size=pool_size)
        sequential_learner = SequentialLearner()
        feature_reduction = FeatureReduction(img_w=img_w, img_h=img_h,
                                             pool_size=pool_size, conv_filters=conv_filters)
        output = Output(output_size)

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_w, img_h, 1)

        inputs = Input(name='the_input', shape=input_shape, dtype='float32')
        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        x = feature_extraction(inputs)
        x = feature_reduction(x)
        x = sequential_learner(x)
        predictions = output(x)

        # CTC loss implemented as a Lambda layer to handle input_length and label_length
        loss_out = Lambda(self._ctc_lambda_func, output_shape=(1,), name='ctc')([predictions, labels, input_length, label_length])

        super(OCRNet, self).__init__(
            inputs=[inputs, labels, input_length, label_length],
            outputs=loss_out,
            name=name,
            **kwargs)

        # Keras function for decoding CTC predictions
        flattened_input_length = K.reshape(input_length, (-1,))
        top_k_decoded, _ = K.ctc_decode(predictions, flattened_input_length)
        self.decoder = K.function([inputs, flattened_input_length], [top_k_decoded[0]])

        # Save submodules so that potential fused wrapper can access them if needed
        self.feature_extraction = feature_extraction
        self.feature_reduction = feature_reduction
        self.sequential_learner = sequential_learner
        self.output_layer = output

    def _ctc_lambda_func(self, args):
        predictions, labels, input_length, label_length = args
        # ignore first two time-steps of predictions as per original implementation comment
        predictions = predictions[:, 2:, :]
        return K.ctc_batch_cost(labels, predictions, input_length, label_length)

# Since the issue is about comparing MirroredStrategy and OneDeviceStrategy implementations,
# we create a fused model that encapsulates two OCRNet models (one for each strategy) and
# compares their outputs to reveal any discrepancy.

class MyModel(tf.keras.Model):
    def __init__(self, output_size=80, img_w=128, img_h=64, max_text_len=32):
        """
        This model wraps two identical OCRNet models (simulating training under different strategies)
        and compares their CTC loss outputs.

        This mimics testing if mirrored strategy and one device strategy give similar output.

        Parameters are defaulted as reasonable guesses inspired by OCR example.
        """
        super(MyModel, self).__init__()
        self.model_one_device = OCRNet(output_size, img_w, img_h, max_text_len, name='OCRNet_OneDevice')
        self.model_mirrored = OCRNet(output_size, img_w, img_h, max_text_len, name='OCRNet_Mirrored')

    def call(self, inputs):
        """
        inputs: tuple or list of four tensors:
            images: (B, img_w, img_h, 1) float32 Tensor
            labels: (B, max_text_len) float32 Tensor
            input_length: (B,1) int64 Tensor
            label_length: (B,1) int64 Tensor

        Outputs a dictionary with:
          - 'loss_one_device': loss from model_one_device (ctc loss)
          - 'loss_mirrored': loss from model_mirrored (ctc loss)
          - 'loss_diff': elementwise absolute difference between the two loss outputs
          - 'loss_equal': boolean tensor indicating if the losses are close within a tolerance (1e-5)
        """
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 4):
            raise ValueError("Input to MyModel must be a list/tuple of four tensors: images, labels, input_length, label_length")

        images, labels, input_length, label_length = inputs

        # Forward pass through both models with identical inputs
        loss_one = self.model_one_device([images, labels, input_length, label_length])
        loss_mirrored = self.model_mirrored([images, labels, input_length, label_length])

        loss_diff = tf.abs(loss_one - loss_mirrored)
        loss_equal = tf.reduce_all(tf.math.less_equal(loss_diff, 1e-5))

        return {
            'loss_one_device': loss_one,
            'loss_mirrored': loss_mirrored,
            'loss_diff': loss_diff,
            'loss_equal': loss_equal
        }

def my_model_function():
    # Return an instance of the fused comparison model with default params
    # Defaults correspond to some commonly used OCR input sizes and max sequence length
    return MyModel(output_size=80, img_w=128, img_h=64, max_text_len=32)

def GetInput():
    """
    Returns synthetic inputs matching the OCRNet and MyModel expectations.

    Assumptions:
    - batch_size = 4 (example)
    - img_w = 128, img_h = 64
    - max_text_len = 32
    - output_size (number of classes) = 80 (e.g., characters plus blank for CTC)

    Output:
    (images, labels, input_length, label_length)
    - images: Tensor float32 (B, img_w, img_h, 1), values in [0,1]
    - labels: Tensor float32 (B, max_text_len), random int class indices or zeros (simulate)
    - input_length: Tensor int64 (B,1), positive numbers representing sequence length <= max_text_len
    - label_length: Tensor int64 (B,1), label lengths <= max_text_len
    """
    batch_size = 4
    img_w, img_h = 128, 64
    max_text_len = 32
    output_size = 80

    # Images: random floats simulating grayscale input images
    images = tf.random.uniform(shape=(batch_size, img_w, img_h, 1), minval=0, maxval=1, dtype=tf.float32)

    # Labels: integer labels to simulate sequences, padded with zeros (float32 to match model input)
    # Labels are class indices in [0, output_size). We simulate by random indices for some positions
    # For padding positions, zeros remain.
    labels_np = tf.random.uniform((batch_size, max_text_len), minval=0, maxval=output_size-1, dtype=tf.int32)
    labels = tf.cast(labels_np, tf.float32)

    # Input length: lengths of input sequences for CTC (must be >= 2 due to prediction truncation in loss)
    # For safety, set between 10 and max_text_len to avoid invalid input lengths
    input_length_np = tf.random.uniform((batch_size, 1), minval=10, maxval=max_text_len, dtype=tf.int32)
    input_length = tf.cast(input_length_np, tf.int64)

    # Label length: length of text labels, between 1 and max_text_len
    label_length_np = tf.random.uniform((batch_size, 1), minval=1, maxval=max_text_len, dtype=tf.int32)
    label_length = tf.cast(label_length_np, tf.int64)

    return (images, labels, input_length, label_length)

