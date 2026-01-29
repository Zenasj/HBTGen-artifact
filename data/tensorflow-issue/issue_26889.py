# tf.random.uniform((B, 256, 256, 3), dtype=tf.float32) ← Input shape inferred from the example inputs and model definition

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base feature detector: Xception without top, pretrained on ImageNet
        self.feature_detector = tf.keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )
        self.feature_detector.trainable = False  # freeze base model
        
        # Classifier layers expanded inline to fix target feeding issue
        # Reflects the original classifier architecture but layers are added directly to MyModel
        self.dense = tf.keras.layers.Dense(256, activation='relu', input_shape=self.feature_detector.output_shape[1:])
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.feature_detector(inputs, training=training)  # feature maps (None,8,8,2048)
        x = self.dense(x)                                    # dense layer applied spatially (None,8,8,256)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.output_dense(x)
        return x

# Custom metrics similar to original user definitions:
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def my_model_function():
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['acc', recall, f1]
    )
    return model

def GetInput():
    # Return a random batch of images matching the expected input shape
    batch_size = 32
    return tf.random.uniform((batch_size, 256, 256, 3), dtype=tf.float32)

