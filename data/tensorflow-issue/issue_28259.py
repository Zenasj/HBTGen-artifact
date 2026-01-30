import tensorflow as tf
import tensorflow.keras as k

mnist = k.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = k.models.Sequential([
	k.layers.Flatten(input_shape=(28, 28)),
	k.layers.Dense(512, activation=tf.nn.elu),
	k.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[k.metrics.Precision()])

model.fit(x_train, y_train, epochs=1)