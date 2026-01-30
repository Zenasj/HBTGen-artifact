from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
from tensorflow.keras.layers import Dense, Input, TimeDistributed
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

def get_model(working_dir, finetune):

    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             alpha=1.4,
                             input_shape=(224, 224, 3),
                             pooling='avg')

    inp = Input((10, 224, 224, 3))
    x = TimeDistributed(base_model)(inp)

    # Comment out the previous two lines
    # and uncomment these next two lines to see the expected behavior

    # inp = base_model.input
    # x = base_model.output

    predictions = Dense(1, activation=sigmoid)(x)

    model = Model(inputs=inp, outputs=predictions)

    if not finetune:
        for layer in base_model.layers:
            layer.trainable = False
        learning_rate = 0.001
    else:
        saved_model_path = os.path.join(working_dir, "saved_model.h5")
        model.load_weights(saved_model_path)
        learning_rate = 0.00001

    binacc = BinaryAccuracy(name="binary_accuracy")
    adam = Adam(lr=learning_rate, epsilon=1e-09, clipnorm=0.001)

    model.compile(optimizer=adam,
                    loss='binary_crossentropy',
                    metrics=[binacc])
    model.summary()

    return model

working = "/content/test"

model = get_model(working, False)
model.save(os.path.join(working, "saved_model.h5"))
model = get_model(working, True)