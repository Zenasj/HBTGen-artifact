from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import scipy

import os
import cv2
from PIL import Image
import numpy as np

# i have habit of writing this on every program, since my GPU memory is only 4GB
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])


SIZE = 224

dataset = []
label = []

parasitized_images = os.listdir('Parasitized/')

for i, image_name in enumerate(parasitized_images):
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread('Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)
    
uninfected_images = os.listdir('Uninfected/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread('Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)
        
dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255     

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def get_model(input_shape = (224, 224, 3)):
    
    with tf.device('/gpu:0'):
    
        vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
        
        for layer in vgg.layers[:-5]:
            print(layer.name)
            layer.trainable = False
            
        x = vgg.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2, activation="softmax")(x)
            
        model = Model(vgg.input, x)
            
        return model



model = get_model(input_shape = (224, 224, 3))

with tf.device('/cpu:0'):
    model.compile(loss="categorical_crossentropy",
                  optimizer = SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=8, epochs=5, validation_data=(X_test, y_test))

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

n=300  #Select the index of image to be loaded for testing
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
print("The prediction for this image is: ", np.argmax(model.predict(input_img)))
print("The actual label for this image is: ", np.argmax(y_test[n]))

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
cm=confusion_matrix(np.argmax(y_test, axis=1), y_pred)  
sns.heatmap(cm, annot=True)

#Identify all images classified as parasitized
parasited_image_idx = np.where(y_pred == 1)[0]

predicted_as_para=[]
for i in parasited_image_idx:
    par_img = X_test[i]
    #plt.imsave("results_classified_as_para/para_"+str(i)+".png", par_img)
    predicted_as_para.append(par_img)
    
predicted_as_para = np.array(predicted_as_para)

from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
cm=confusion_matrix(np.argmax(y_test, axis=1), y_pred)  
sns.heatmap(cm, annot=True)