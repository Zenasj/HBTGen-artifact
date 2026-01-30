from tensorflow.keras import layers

from tensorflow import keras

(x, y), _ = keras.datasets.cifar10.load_data()
x = (x / 255.0).reshape(x.shape[0], -1)

model = keras.Sequential([keras.layers.Dense(10)])

# commenting out these lines result in way slower training
x = x[0:10]
y = y[0:10]

model.compile("sgd", "sparse_categorical_crossentropy")
model.fit(x, y, epochs=1, batch_size=1, steps_per_epoch=10)