from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf

x, y = make_classification(random_state=42)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(20,)),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=50)
model.evaluate(test_x, test_y)
model.save("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)