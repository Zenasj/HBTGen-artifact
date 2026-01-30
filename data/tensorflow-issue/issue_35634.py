import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from numpy.random import RandomState
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

N_CLASSES, N_SAMPLES = 5, 500

for seed in [1, 2]:
    print(f"=== Random seed is {seed} ===")

    tf.random.set_seed(seed)
    rng = RandomState(seed)

    x_train = rng.standard_normal(size=(N_SAMPLES, 10))
    x_test = rng.standard_normal(size=(N_SAMPLES, 10))
    y_train = rng.random_integers(N_CLASSES, size=N_SAMPLES)
    y_test = rng.random_integers(N_CLASSES, size=N_SAMPLES)

    model = Sequential([Dense(32), Dense(N_CLASSES)])
    model.compile("adam", "categorical_crossentropy", ["accuracy"])
    early_stopping = EarlyStopping(
        "val_accuracy", patience=10, verbose=1, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        callbacks=[early_stopping],
        verbose=0,
        validation_data=(x_test, y_test),
    )
    best_acc = max(history.history["val_accuracy"])
    _, eval_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Best accuracy in training: {best_acc}. In evaluation: {eval_acc}\n")