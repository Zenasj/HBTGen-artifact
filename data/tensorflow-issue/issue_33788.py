from tensorflow.keras import layers
from tensorflow.keras import models

# Importing the Keras libraries and packages
from keras.models import Sequential, save_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np

imageResX, imageResY = 256, 256

def CNNmodel():
	classifier = Sequential()
	classifier.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(imageResX, imageResY, 3)))
	classifier.add(MaxPooling2D(pool_size=(3,3)))
	classifier.add(Flatten())
	classifier.add(Dense(units=128, activation='relu'))
	classifier.add(Dropout(rate=0.5))
	classifier.add(Dense(units=4, activation='softmax'))
	classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	return classifier

# create a data generators
train_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255) 

training_set = train_datagen.flow_from_directory('MY DATA/Training', target_size = (imageResX, imageResY), batch_size = 64, class_mode = 'categorical')
evaluate_set = train_datagen.flow_from_directory('MY DATA/Evaluation', target_size = (imageResX, imageResY), batch_size = 64, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('MY DATA/Testing', target_size = (imageResX, imageResY), batch_size = 64, class_mode = 'categorical', shuffle=False)

step_size_train=training_set.n//training_set.batch_size
step_size_evaluate=evaluate_set.n//evaluate_set.batch_size
step_size_test=test_set.n//test_set.batch_size

model = CNNmodel()
history = model.fit_generator(generator=training_set, steps_per_epoch=step_size_train, epochs=1, validation_data=evaluate_set, validation_steps=step_size_evaluate)

labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())

# Save KERAS model
modelName = "ST-AI-Model"
save_model(model, str(modelName+".h5"))

# Convert KERAS model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(str(modelName+".tflite"), "wb").write(tflite_model)

prediction = model.predict_generator(generator=test_set, verbose=1)
classes = test_set.classes[test_set.index_array]
predicted_class_indices = np.argmax(prediction, axis=1)
target_names = [labels[k] for k in range(len(training_set.class_indices))]


print('Confusion Matrix')
print(confusion_matrix(test_set.classes[test_set.index_array], predicted_class_indices))
print('Classification Report')
print(sum(predicted_class_indices==classes)/len(test_set.classes))
print(classification_report(test_set.classes[test_set.index_array], predicted_class_indices, target_names=target_names))

# Importing the Keras libraries and packages
from keras.models import Sequential, save_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf

imageResX, imageResY = 256, 256

def CNNmodel():
	classifier = Sequential()
	classifier.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(imageResX, imageResY, 3)))
	classifier.add(MaxPooling2D(pool_size=(3,3)))
	classifier.add(Flatten())
	classifier.add(Dense(units=128, activation='relu'))
	classifier.add(Dropout(rate=0.5))
	classifier.add(Dense(units=4, activation='softmax'))
	classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	return classifier

model = CNNmodel()

# Convert KERAS model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("test.tflite", "wb").write(tflite_model)

import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D