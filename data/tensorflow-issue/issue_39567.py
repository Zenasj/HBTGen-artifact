import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

inputs = tf.keras.layers.Input(shape=[224, 224, 3])

def Net(inputs):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
    output = base_model.get_layer("pool3_conv").output
    x = Conv2D(128, 3, activation='relu', padding='same')(output)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name='clf_output')(x)

    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[x])

    return model

def create_ensemble(models, inputs):
    for i in range(len(models)):
        # Each model is a Net object
        model = models[i]
        for layer in model.layers[1:]:
            layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer._name

    stack_outputs = [model(inputs) for model in models]
    merge = Concatenate()(stack_outputs) 
    x = Dense(16, activation='relu')(merge)
    x = Dense(2, activation='softmax')(x)

    print(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=x, name='ensemble')

    return model