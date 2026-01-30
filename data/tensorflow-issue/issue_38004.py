from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 

## --- LOAD DATA ---
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

encoder = sklearn.preprocessing.OneHotEncoder(dtype=np.float32)
encoder.fit(Y_train.reshape((-1, 1)))

Y_train_encoded = encoder.transform(Y_train.reshape((-1, 1))).toarray()
Y_test_encoded = encoder.transform(Y_test.reshape((-1, 1))).toarray()


## --- SETUP MODEL ---
def setupModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,28)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()]
                 )
    return model

# this model now uses the sklearn API instead of the Keras API
# it still trains using tensorflow under the hood
model = KerasClassifier(build_fn=setupModel,
                        epochs=2, # show that training executes
                        batch_size=256,
                       )

# train the model
model.fit(X_train, Y_train_encoded)

# evaluate the test accuracy
test_accuracy = model.score(X_test, Y_test_encoded, verbose=0)
print(f"The test accuracy is {test_accuracy * 100:.2f}%")

import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 

## --- LOAD DATA ---
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

encoder = sklearn.preprocessing.OneHotEncoder(dtype=np.float32)
encoder.fit(Y_train.reshape((-1, 1)))

Y_train_encoded = encoder.transform(Y_train.reshape((-1, 1))).toarray()
Y_test_encoded = encoder.transform(Y_test.reshape((-1, 1))).toarray()


## --- SETUP MODEL ---
def setupModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,28)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy()]
                 )
    return model

# this model now uses the sklearn API instead of the Keras API
# it still trains using tensorflow under the hood
model = KerasClassifier(build_fn=setupModel,
                        epochs=2, # show that training executes
                        batch_size=256,
                       )

# train the model
model.fit(X_train, Y_train_encoded)

# evaluate the test accuracy
test_accuracy = model.score(X_test, Y_test_encoded, verbose=0)
print(f"The test accuracy is {test_accuracy * 100:.2f}%")

Y_pred = model.model.predict(X_test)
acc = tf.keras.metrics.CategoricalAccuracy()
acc.update_state(Y_test, Y_pred)
print(f"The (true) categorical accuracy is {acc.result().numpy()*100:.2f}%")

import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 

## --- LOAD DATA ---
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

encoder = sklearn.preprocessing.OneHotEncoder(dtype=np.float32)
encoder.fit(Y_train.reshape((-1, 1)))

Y_train_encoded = encoder.transform(Y_train.reshape((-1, 1))).toarray()
Y_test_encoded = encoder.transform(Y_test.reshape((-1, 1))).toarray()


## --- SETUP MODEL ---
def setupModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,28)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics = ['accuracy'] #metrics=[tf.keras.metrics.CategoricalAccuracy()]
                 )
    return model

# this model now uses the sklearn API instead of the Keras API
# it still trains using tensorflow under the hood
model = KerasClassifier(build_fn=setupModel,
                        epochs=2, # show that training executes
                        batch_size=256,
                       )

# train the model
model.fit(X_train, Y_train_encoded)

# evaluate the test accuracy
test_accuracy = model.score(X_test, Y_test_encoded, verbose=0)
print(f"The test accuracy is {test_accuracy * 100:.2f}%")

# test case to compare metrics
Y_pred = model.model.predict(X_test)
acc = tf.keras.metrics.CategoricalAccuracy()
acc.update_state(Y_test, Y_pred)
print(f"The (true) categorical accuracy is {acc.result().numpy()*100:.2f}%")