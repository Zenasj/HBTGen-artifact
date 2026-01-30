from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from itertools import chain
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow

# tensorboard --logdir=logs/ --host localhost --port 8088

seed = 42
np.random.seed = seed

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 256        #Resize images (height  = X, width = Y)
SIZE_Y = 256
CHANNELS = 3


# TRAIN_DATA= 'SegTrackv2_Dataset/SegTrackv2/Train/Images'
# TEST_DATA = 'SegTrackv2_Dataset/SegTrackv2/Test/Images'

# TRAIN_DATA= 'Train/Images'
# TEST_DATA = 'Test/Images'

TRAIN_DATA= 'PatchTrain/Images'
TEST_DATA = 'PatchTest/Images'

train_ids = next(os.walk(TRAIN_DATA))[1]
test_ids = next(os.walk(TEST_DATA))[1]
classnames = test_ids

train_count = 0
test_count = 0
mask_count = 0

train_img_list = []
mask_img_list = []
test_img_list = []

# Get the list of all images in the training input and mask subfolders 

for names in train_ids:
    train_path = (os.path.join(f'{TRAIN_DATA}/{names}/Images'))
    train_img = glob.glob(f'{train_path}/**.png')
    train_img.extend(glob.glob(f'{train_path}/**.bmp'))
    train_img_list = list(chain(train_img, train_img_list))
    train_count = len(train_img_list)


    mask_path = (os.path.join(f'{TRAIN_DATA}/{names}/Masks'))
    mask_img = glob.glob(f'{mask_path}/**.png')
    mask_img.extend(glob.glob(f'{mask_path}/**.bmp'))
    mask_img_list = list(chain(mask_img, mask_img_list))
    mask_count =  len(mask_img_list)


    test_path = (os.path.join(f'{TEST_DATA}/{names}/Images'))
    test_img = glob.glob(f'{test_path}/**.png')
    test_img.extend(glob.glob(f'{test_path}/**.bmp'))
    test_img_list = list(chain(test_img, test_img_list))
    test_count = len(test_img_list)

# Create numpy array to store the images

X_train = np.zeros((train_count, SIZE_X, SIZE_Y,CHANNELS), dtype=np.uint8)
Y_train = np.zeros((train_count, SIZE_X, SIZE_Y), dtype=bool)    # Mask is the boolian value of segmented Images

# Resize Images and masks to the desired input size of convolutional layer

print('Resizing training images and masks')
for i in tqdm(range((100))): # train_count
    
    img = cv2.imread(train_img_list[i], cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    X_train[i] = img                # Fill empty X_train with values from img
       
    
    mask = cv2.imread(mask_img_list[i], 0)       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X))    
    Y_train[i] = mask 
     
# Resize test images

X_test = np.zeros((test_count, SIZE_X, SIZE_Y,CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for t in tqdm(range(50)): # test_count
    
    img = cv2.imread(test_img_list[t],cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    X_test[t] = img

print('Done!')

X_VAL = X_train[int(X_train.shape[0]*0.8):]
Y_VAL = Y_train[int(X_train.shape[0]*0.8):]

# Plot some random Training images and their corresponding masks

for i in range(1,7,2):
    ix = random.randint(0, train_count)
    plt.subplot(3,2,i)
    plt.title('X_train',fontsize = 5)
    plt.imshow(X_train[ix])
    plt.imshow(Y_train[ix],cmap='jet', alpha=0.5)

    plt.subplot(3,2,i+1)
    plt.title('Y_train',fontsize = 5)
    plt.imshow(np.squeeze(Y_train[ix]))


plt.show()


#Build the model

inputs = tf.keras.layers.Input((SIZE_X, SIZE_Y, CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
####################################################################

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


####################################################################
#Modelcheckpoint

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), # Patience = 3
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=64, epochs=10, callbacks=callbacks) # val_split = 0.1

#####################################################################


idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.8)], verbose=1)
preds_val = model.predict(X_VAL, verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t =   (preds_val   > 0.5).astype(np.uint8)
preds_test_t =  (preds_test  > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples


for i in range(1,10,3):
    
    ix = random.randint(0, len(preds_train_t))
    plt.subplot(3,3,i)
    plt.title('X_train',fontsize = 5)
    plt.imshow(X_train[ix])
    plt.imshow(np.squeeze(preds_train_t[ix]),cmap = 'jet',alpha=0.5)

    plt.subplot(3,3,i+1)
    plt.title('Y_train',fontsize = 5)
    plt.imshow(Y_train[ix])

    plt.subplot(3,3,i+2)
    plt.title('Predict_Train',fontsize = 5)
    plt.imshow(np.squeeze(preds_train_t[ix]))      

plt.show()

# Perform a sanity check on some random Validation samples


for i in range(1,10,3):
    ix = random.randint(0, len(preds_val_t))
    
    plt.subplot(3,3,i)
    plt.title('X_val',fontsize = 5)
    plt.imshow(X_VAL[ix])
    plt.imshow(np.squeeze(preds_val_t[ix]),cmap = 'jet',alpha=0.5)

    plt.subplot(3,3,i+1)
    plt.title('Y_val',fontsize = 5)
    plt.imshow(Y_VAL[ix])

    plt.subplot(3,3,i+2)
    plt.title('Predict_Val',fontsize = 5)
    plt.imshow(np.squeeze(preds_val_t[ix]))
   
plt.show()

# Model Prediction on test Images 
for j in range(3):
    for i in range(1,7,2):
        
        ix = random.randint(0, len(preds_test_t))
        
        plt.subplot(3,2,i)
        plt.title('X_test',fontsize = 5)
        plt.imshow(X_test[ix])
        plt.imshow(np.squeeze(preds_test_t[ix]),cmap = 'jet',alpha=0.5)

        plt.subplot(3,2,i+1)
        plt.title('Predict_Test',fontsize = 5)
        plt.imshow(np.squeeze(preds_test_t[ix]))
        
    plt.show()

print('Done!')