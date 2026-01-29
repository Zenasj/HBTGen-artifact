# tf.random.uniform((B, 120, 100, 250, 1), dtype=tf.float32) ← Assumed input shape: batch dim unknown, 120 frames, height=100, width=250, channels=1 (grayscale cropped mouth images)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=5):
        super().__init__()
        # Following the original LipReadingModel's architecture inferred from the issue
        
        # Conv3D layers with relu and maxpool3d
        self.conv1 = layers.Conv3D(128, kernel_size=3, padding='same', input_shape=(120, 100, 250, 1))
        self.act1 = layers.Activation('relu')
        self.pool1 = layers.MaxPool3D(pool_size=(1, 2, 2))
        
        self.conv2 = layers.Conv3D(256, kernel_size=3, padding='same')
        self.act2 = layers.Activation('relu')
        self.pool2 = layers.MaxPool3D(pool_size=(1, 2, 2))
        
        self.conv3 = layers.Conv3D(120, kernel_size=3, padding='same')
        self.act3 = layers.Activation('relu')
        self.pool3 = layers.MaxPool3D(pool_size=(1, 2, 2))
        
        # TimeDistributed Flatten to collapse spatial dims per frame
        self.td_flatten = layers.TimeDistributed(layers.Flatten())
        
        # Bidirectional LSTM layers with dropout
        self.bi_lstm1 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, kernel_initializer='orthogonal'))
        self.dropout1 = layers.Dropout(0.5)
        
        self.bi_lstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, kernel_initializer='orthogonal'))
        self.dropout2 = layers.Dropout(0.5)
        
        # Final Dense layer with softmax activation for classification (+1 for CTC blank or similar)
        self.dense = layers.Dense(num_classes + 1, activation='softmax', kernel_initializer='he_normal')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        x = self.td_flatten(x)
        
        x = self.bi_lstm1(x)
        if training:
            x = self.dropout1(x, training=training)
        else:
            x = self.dropout1(x)
        
        x = self.bi_lstm2(x)
        if training:
            x = self.dropout2(x, training=training)
        else:
            x = self.dropout2(x)
        
        x = self.dense(x)
        return x

def my_model_function():
    # Construct MyModel with default 5 classes (A, U, EE, E, space) as per the char_to_num vocab
    return MyModel(num_classes=5)

def GetInput():
    # Create a random float32 tensor input with the expected shape:
    # Batch size assumed 2 to accommodate augmentor doubling batch with flipping concat (as per DatasetPreparer)
    # Frames=120 (from dataset padded shapes), height=100, width=250 (mouth crop shape), channels=1 grayscale
    # Values normalized roughly -1 to 1 by (data - mean)/std in loader, so let's generate standard normal for generality.
    return tf.random.uniform(shape=(2, 120, 100, 250, 1), dtype=tf.float32)

