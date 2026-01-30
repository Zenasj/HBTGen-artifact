import random
from tensorflow.keras import layers
from tensorflow.keras import models

import json
import numpy as np
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Masking, Input, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

y_raw = np.random.randint(19, size=(35000,200))
y = to_categorical(y_raw, num_classes=19)
X = np.random.rand(35000, 200, 311)

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(200,311)))
model.add(
            Bidirectional(
                        GRU(
                            256,
                            return_sequences=True,
                            unroll=True,
                            recurrent_dropout=0.233,
                            recurrent_activation="sigmoid"
                           )
                         )
         )

model.add(TimeDistributed(Dense(19, activation="softmax")))

model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["categorical_accuracy"],
            )

model.summary()

architecture_path = "candidate_architecture.json"
model_json = model.to_json()
with open(architecture_path, 'w') as json_file:
    json_file.write(model_json)
print(f"Saved model architecture to {architecture_path}")

filepath = "candidate_model-{epoch:02d}-{val_categorical_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
callbacks_list = [checkpoint]

history = model.fit(X, y, epochs=10,
                    validation_split=0.2,
                    batch_size=16,
                    callbacks=callbacks_list)