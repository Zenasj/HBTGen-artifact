# tf.random.uniform((B, 32, 320, 1), dtype=tf.float32) ← input shape inferred from model input and OCR dataset preprocessing

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.layers import Lambda, Bidirectional, Permute, TimeDistributed, Flatten, LSTM, Dense
import numpy as np

# Define CTC loss as a layer function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # CNN blocks inspired by VGG-like architecture (5 conv blocks)
        self.block1_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.block1_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.block1_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.block2_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.block2_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        self.block3_conv2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        self.block3_conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        self.block3_conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')
        self.block3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.block4_conv2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.block4_conv3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        self.block4_conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')
        self.block4_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        self.block5_conv2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        self.block5_conv3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        self.block5_conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')
        self.block5_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.permute = Permute((2, 1, 3), name='permute')

        # Convert each timestep to vector with TimeDistributed Flatten
        self.timedistrib = TimeDistributed(Flatten(), name='timedistrib')

        # Two layers of bidirectional LSTM with relu dense in between
        self.bilstm1 = Bidirectional(LSTM(512, return_sequences=True), name='bidirectional_1')
        self.dense = layers.Dense(512, activation='relu', name='dense')
        self.bilstm2 = Bidirectional(LSTM(512, return_sequences=True), name='bidirectional_2')

        # Output dense layer with softmax over 21713 classes (Chinese characters + special tokens)
        self.out_dense = layers.Dense(21713, activation='softmax', name='orc_out')

    def call(self, inputs, training=False):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_conv4(x)
        x = self.block3_pool(x)

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_conv4(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_conv4(x)
        x = self.block5_pool(x)

        x = self.permute(x)
        x = self.timedistrib(x)

        x = self.bilstm1(x)
        x = self.dense(x)
        x = self.bilstm2(x)

        out = self.out_dense(x)
        return out


def my_model_function():
    '''
    Returns a tuple:
    - training_model: Keras Model that accepts 4 inputs and outputs CTC loss (for training)
    - predict_model: Keras Model that takes image input and outputs predicted probabilities (for inference)
    
    For fitting, the 4 inputs are:
    (images, labels, input_length, label_length)
    
    The training model outputs the CTC loss directly as 'ctc' output.
    '''
    # OCR model inputs
    image_input = Input(shape=(32, 320, 1), name='image_input', dtype=tf.float32)  # Batch size is None
    labels = Input(name='the_labels', shape=[None], dtype=tf.int32)  # Sparse labels indices, variable length
    input_length = Input(name='input_length', shape=[1], dtype=tf.int32)
    label_length = Input(name='label_length', shape=[1], dtype=tf.int32)

    base_model = MyModel()
    y_pred = base_model(image_input)  # (batch, time_steps, num_classes)

    # Define CTC loss as Lambda layer output
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # Model for training with CTC loss
    training_model = Model(inputs=[image_input, labels, input_length, label_length], outputs=loss_out)

    # Compile with dummy loss because loss is calculated in Lambda layer
    training_model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})

    # Model for inference to predict text probabilities from image only
    predict_model = Model(inputs=image_input, outputs=y_pred)

    # Attach base_model to training_model for access if needed
    training_model.base_model = base_model

    return training_model, predict_model


def GetInput():
    '''
    Returns a batch of inputs as expected by the training model:
    - images: float32 tensor shape (B, 32, 320, 1) normalized between 0 and 1
    - labels: int32 tensor shape (B, None) with padded label sequences (here randomized for example)
    - input_length: int32 tensor (B, 1) representing length of inputs to CTC (after CNN + pooling)
    - label_length: int32 tensor (B, 1) representing length of labels

    Assumptions:
    - Batch size B=4 chosen arbitrarily for example
    - Time steps after conv + pooling: original width 320 reduced by factor of 2^5=32 (5 max poolings with stride 2 each)
      => Time steps = width//32 = 320//32=10 (matches output time dimension in model summary)
    - Labels length arbitrary <=10 for example

    This input can be directly fed to model.fit for demonstration/testing.
    '''
    B = 4
    H = 32
    W = 320
    C = 1

    # Random float images normalized
    images = tf.random.uniform((B, H, W, C), minval=0, maxval=1, dtype=tf.float32)

    # Labels: padded integer sequences, values in range [0,21712]
    max_label_length = 10
    labels = tf.random.uniform((B, max_label_length), minval=0, maxval=21713, dtype=tf.int32)

    # Input sequence length after CNN+pooling layers is 10
    input_length = tf.ones((B, 1), dtype=tf.int32) * 10

    # Label lengths (random positive integers up to max_label_length)
    label_length = tf.random.uniform((B, 1), minval=1, maxval=max_label_length + 1, dtype=tf.int32)

    return [images, labels, input_length, label_length]

