# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32) ← assuming input shape from vgg16_model in the issue

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD

class MyModel(tf.keras.Model):
    def __init__(self, in_shape=(128, 128, 3), out_shape=17, beta=2):
        super().__init__()
        # Load VGG16 base model without top layers
        # Freeze all layers first
        base_model = VGG16(include_top=False, input_shape=in_shape)
        for layer in base_model.layers:
            layer.trainable = False
        
        # Allow last VGG block layers to be trainable
        # Names based on VGG16 architecture
        try:
            base_model.get_layer('block5_conv1').trainable = True
            base_model.get_layer('block5_conv2').trainable = True
            base_model.get_layer('block5_conv3').trainable = True
            base_model.get_layer('block5_pool').trainable = True
        except ValueError:
            # If layers not found for some reason, pass silently
            pass
        
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
        output = Dense(out_shape, activation='sigmoid')(x)
        
        # Final model inside this class
        self.model = Model(inputs=base_model.inputs, outputs=output)
        self.beta = beta

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def fbeta(self, y_true, y_pred):
        """
        Custom fbeta metric compatible with keras and tf.function.
        Computes the fbeta score (default beta=2).
        """
        beta = self.beta
        
        y_pred = K.clip(y_pred, 0, 1)
        # Round predictions and clips to 0 or 1
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)
        
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
        
        fbeta_score = K.mean((1 + beta ** 2) * (p * r) / ((beta ** 2) * p + r + K.epsilon()))
        return fbeta_score

def my_model_function():
    """
    Returns an instance of MyModel with default input and output shapes,
    ready for compilation or inference.
    """
    model_instance = MyModel()
    # Compile model with SGD optimizer and binary_crossentropy loss, and fbeta metric
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model_instance.model.compile(optimizer=opt, 
                               loss='binary_crossentropy', 
                               metrics=[model_instance.fbeta])
    return model_instance

def GetInput():
    """
    Returns a random tensor of shape (1, 128, 128, 3) with float32 dtype,
    suitable as input to MyModel.
    Batch size 1 is assumed as typical for prediction.
    """
    # Generate a batch with a single random image tensor
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

