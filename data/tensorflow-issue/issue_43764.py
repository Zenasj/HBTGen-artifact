from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
nb_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# oversimplified model just for example
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

loss_func = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

for i, (xx, yy) in enumerate(dataset):
    with tf.GradientTape() as tape:
        y_pred = model(xx)
        loss = loss_func(yy, y_pred)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    print('batch {} processed'.format(i))

# if used this approach to train - everything works
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam())
# model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test))
# model.evaluate(x_test, y_test, batch_size=128, verbose=1)