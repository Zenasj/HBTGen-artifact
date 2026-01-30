import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import pandas as pd
import numpy as np

print("TensorFlow version : ", tf.__version__)

DIRECTIONS = [
    "in",
    "out",
]

SAMPLES_PER_DIRECTION = 20

NUM_DIRECTIONS = len(DIRECTIONS)

ONE_HOT_ENCODED_DIRECTIONS = np.eye(NUM_DIRECTIONS)

inputs = []
outputs = []

# read each csv file and push an input and output
for direction_index in range(NUM_DIRECTIONS):
    direction = DIRECTIONS[direction_index]
    print(f"Processing index {direction_index} for direction '{direction}'.")

    output = ONE_HOT_ENCODED_DIRECTIONS[direction_index]

    df = pd.read_csv("input/" + direction + ".csv")

    # calculate the number of gesture recordings in the file
    num_recordings = int(df.shape[0] / SAMPLES_PER_DIRECTION)

    print(f"\tThere are {num_recordings} recordings of the {direction} direction.")

    print(print(df))

    for i in range(num_recordings):
        tensor = []
        for j in range(SAMPLES_PER_DIRECTION):
            index = i * SAMPLES_PER_DIRECTION + j

            tensor += [
                (df['p1'][index]),
                (df['p2'][index]),
                (df['p3'][index]),
                (df['p4'][index]),
                (df['p5'][index]),
                (df['p6'][index]),
                (df['p7'][index]),
                (df['p8'][index]),
                (df['p9'][index]),
                (df['p10'][index]),
                (df['p11'][index]),
                (df['p12'][index]),
                (df['p13'][index]),
                (df['p14'][index]),
                (df['p15'][index]),
                (df['p16'][index]),
                (df['p17'][index]),
                (df['p18'][index]),
                (df['p19'][index]),
                (df['p20'][index]),
                (df['p21'][index]),
                (df['p22'][index]),
                (df['p23'][index]),
                (df['p24'][index]),
                (df['p25'][index]),
                (df['p26'][index]),
                (df['p27'][index]),
                (df['p28'][index]),
                (df['p29'][index]),
                (df['p30'][index]),
                (df['p31'][index]),
                (df['p32'][index]),
                (df['p33'][index]),
                (df['p34'][index]),
                (df['p35'][index]),
                (df['p36'][index]),
                (df['p37'][index]),
                (df['p38'][index]),
                (df['p39'][index]),
                (df['p40'][index]),
                (df['p41'][index]),
                (df['p42'][index]),
                (df['p43'][index]),
                (df['p44'][index]),
                (df['p45'][index]),
                (df['p46'][index]),
                (df['p47'][index]),
                (df['p48'][index]),
                (df['p49'][index]),
                (df['p51'][index]),
                (df['p52'][index]),
                (df['p53'][index]),
                (df['p54'][index]),
                (df['p55'][index]),
                (df['p56'][index]),
                (df['p57'][index]),
                (df['p58'][index]),
                (df['p59'][index]),
                (df['p61'][index]),
                (df['p62'][index]),
                (df['p63'][index]),
                (df['p64'][index])
            ]

        inputs.append(tensor)
        outputs.append(output)

# convert the list to numpy array
inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")

num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]

TRAIN_SPLIT = int(0.8 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(15, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(inputs, outputs, epochs=100)

predictions = model.predict(inputs_test)

# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
print("actual =\n", outputs_test)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # , tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
tfmodel = converter.convert()
open("model.tflite","wb").write(tfmodel)