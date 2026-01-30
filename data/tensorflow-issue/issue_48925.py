from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

INIT_LR = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8

basemodel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))
basemodel.trainable = False

# MobileNetV2
flatten = basemodel.output
bboxHead = Flatten(name="flatten")(flatten)
bboxHead = Dense(128, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

model = Model(inputs=basemodel.input, outputs=bboxHead)
opt = Adam(INIT_LR)
model.compile(loss="mse", optimizer=opt)
H = model.fit(
        trainImages, trainTargets, 
        validation_data=(testImages, testTargets),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1)

import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(model)
q_aware_model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])