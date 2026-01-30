import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import psutil

import numpy as np
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def dummy_model(n_classes, n_features, seq_length):
    input = Input(shape=(seq_length, n_features))
    main = Bidirectional(LSTM(128, return_sequences=False))(input)

    prediction = Dense(n_classes, activation="softmax")(main)
    optimiser = Adam(lr=1e-3)
    model = Model(inputs=[input], outputs=prediction)
    model.compile(optimiser, "categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_model():
    x_train = np.random.random_sample((1000, 50, 100))
    x_train = x_train.astype(np.float32)
    y_train = np.zeros((1000, 10), dtype=np.float32)
    model = dummy_model(n_classes=10, n_features=100, seq_length=50)
    model.fit(
        x_train, y_train, epochs=1,
        validation_split=0.1, batch_size=64,
        verbose=0
    )


if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    n = 20

    for i in range(n):
        fit_model()
        print(f"#--- Run {i + 1} of {n} memory used (MB): {process.memory_info().rss / 1e6}")

#--- Run 1 of 20 memory used (MB): 731.81184
#--- Run 2 of 20 memory used (MB): 991.137792
#--- Run 3 of 20 memory used (MB): 1027.985408
#--- Run 4 of 20 memory used (MB): 1089.724416
#--- Run 5 of 20 memory used (MB): 1127.616512
#--- Run 6 of 20 memory used (MB): 1165.471744
#--- Run 7 of 20 memory used (MB): 1211.96544
#--- Run 8 of 20 memory used (MB): 1247.653888
#--- Run 9 of 20 memory used (MB): 1272.410112
#--- Run 10 of 20 memory used (MB): 1289.478144
#--- Run 11 of 20 memory used (MB): 1298.57536
#--- Run 12 of 20 memory used (MB): 1317.429248
#--- Run 13 of 20 memory used (MB): 1346.781184
#--- Run 14 of 20 memory used (MB): 1369.145344
#--- Run 15 of 20 memory used (MB): 1402.55232
#--- Run 16 of 20 memory used (MB): 1409.634304
#--- Run 17 of 20 memory used (MB): 1413.599232
#--- Run 18 of 20 memory used (MB): 1419.091968
#--- Run 19 of 20 memory used (MB): 1450.14784
#--- Run 20 of 20 memory used (MB): 1468.604416

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(5)

#--- Run 1 of 20 memory used (MB): 578.285568
#--- Run 2 of 20 memory used (MB): 603.820032
#--- Run 3 of 20 memory used (MB): 636.522496
#--- Run 4 of 20 memory used (MB): 648.904704
#--- Run 5 of 20 memory used (MB): 659.59936
#--- Run 6 of 20 memory used (MB): 669.63456
#--- Run 7 of 20 memory used (MB): 674.701312
#--- Run 8 of 20 memory used (MB): 685.817856
#--- Run 9 of 20 memory used (MB): 692.137984
#--- Run 10 of 20 memory used (MB): 694.984704
#--- Run 11 of 20 memory used (MB): 706.154496
#--- Run 12 of 20 memory used (MB): 709.722112
#--- Run 13 of 20 memory used (MB): 714.248192
#--- Run 14 of 20 memory used (MB): 717.27104
#--- Run 15 of 20 memory used (MB): 718.78656
#--- Run 16 of 20 memory used (MB): 721.498112
#--- Run 17 of 20 memory used (MB): 723.771392
#--- Run 18 of 20 memory used (MB): 724.639744
#--- Run 19 of 20 memory used (MB): 727.764992
#--- Run 20 of 20 memory used (MB): 729.41568