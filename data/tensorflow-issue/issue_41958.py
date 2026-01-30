from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Create first model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

model.save('saved_model/my_model')

# Create second model (load the saved model)
new_model = tf.keras.models.load_model('saved_model/my_model')

# First model's results
loss1, acc1 = model.evaluate(test_images, test_labels)
accuracy_score1 = accuracy_score(test_labels, np.argmax(model.predict(test_images), axis=1))
# loss1, acc1 = [0.4392167329788208, 0.8629999756813049]
# accuracy_score1 = 0.863

# Second model's results
loss2, acc2 = new_model.evaluate(test_images, test_labels)                           
accuracy_score2 = accuracy_score(test_labels, np.argmax(new_model.predict(test_images), axis=1))
# loss2, acc2 = [0.4392167329788208, 0.0860000029206276] <- THIS!!!
# accuracy_score2 = 0.863