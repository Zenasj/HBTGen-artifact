from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

# Set the input shape of the images (adjust based on the input image size)
input_shape = (128, 128, 3)  # Adjust based on the input image size

# Set the number of segmentation classes 
n_classes = 1  # Number of segmentation classes

# Define the model architecture
inputs = Input(shape=input_shape)  # Define the input layer with the specified input shape

# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # First convolutional layer with 64 filters
conv1 = BatchNormalization()(conv1)  # Apply batch normalization to normalize the activations
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)  # Second convolutional layer with 64 filters
conv1 = BatchNormalization()(conv1)  # Apply batch normalization to normalize the activations
pool1 = MaxPooling2D((2, 2))(conv1)  # Max pooling layer with a pool size of (2, 2)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)  # Convolutional layer with 128 filters
conv2 = BatchNormalization()(conv2)  # Apply batch normalization to normalize the activations
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)  # Convolutional layer with 128 filters
conv2 = BatchNormalization()(conv2)  # Apply batch normalization to normalize the activations
pool2 = MaxPooling2D((2, 2))(conv2)  # Max pooling layer with a pool size of (2, 2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)  # Convolutional layer with 256 filters
conv3 = BatchNormalization()(conv3)  # Apply batch normalization to normalize the activations
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # Convolutional layer with 256 filters
conv3 = BatchNormalization()(conv3)  # Apply batch normalization to normalize the activations
pool3 = MaxPooling2D((2, 2))(conv3)  # Max pooling layer with a pool size of (2, 2)

conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)  # Convolutional layer with 512 filters
conv4 = BatchNormalization()(conv4)  # Apply batch normalization to normalize the activations
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)  # Convolutional layer with 512 filters
conv4 = BatchNormalization()(conv4)  # Apply batch normalization to normalize the activations
drop4 = Dropout(0.5)(conv4)  # Apply dropout regularization with a rate of 0.5
pool4 = MaxPooling2D((2, 2))(drop4)  # Max pooling layer with a pool size of (2, 2)

conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)  # Convolutional layer with 1024 filters
conv5 = BatchNormalization()(conv5)  # Apply batch normalization to normalize the activations
conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)  # Convolutional layer with 1024 filters
conv5 = BatchNormalization()(conv5)  # Apply batch normalization to normalize the activations
drop5 = Dropout(0.5)(conv5)  # Apply dropout regularization with a rate of 0.5

# Decoder
up6 = concatenate([UpSampling2D((2, 2))(drop5), conv4], axis=-1)  # Upsampling layer with a scale factor of (2, 2)
conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)  # Convolutional layer with 512 filters
conv6 = BatchNormalization()(conv6)  # Apply batch normalization to normalize the activations
conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)  # Convolutional layer with 512 filters
conv6 = BatchNormalization()(conv6)  # Apply batch normalization to normalize the activations

up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=-1)  # Upsampling layer with a scale factor of (2, 2)
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)  # Convolutional layer with 256 filters
conv7 = BatchNormalization()(conv7)  # Apply batch normalization to normalize the activations
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)  # Convolutional layer with 256 filters
conv7 = BatchNormalization()(conv7)  # Apply batch normalization to normalize the activations

up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=-1)  # Upsampling layer with a scale factor of (2, 2)
conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)  # Convolutional layer with 128 filters
conv8 = BatchNormalization()(conv8)  # Apply batch normalization to normalize the activations
conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)  # Convolutional layer with 128 filters
conv8 = BatchNormalization()(conv8)  # Apply batch normalization to normalize the activations

up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=-1)  # Upsampling layer with a scale factor of (2, 2)
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)  # Convolutional layer with 64 filters
conv9 = BatchNormalization()(conv9)  # Apply batch normalization to normalize the activations
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)  # Convolutional layer with 64 filters
conv9 = BatchNormalization()(conv9)  # Apply batch normalization to normalize the activations

outputs = Conv2D(n_classes, (1, 1), activation='softmax')(conv9)  # Convolutional layer for output

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Print the model summary
model.summary()


# Set the optimizer for the model
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# Compile the model with loss function and metrics
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

images_path = '/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/images/val2014'
masks_path = '/kaggle/working/mask_val_2014'
batch_size = 8

val_generator = CustomDataGenerator(images_path, masks_path, batch_size)


# Fit the model with the training generator
train_steps =  len(os.listdir( "/kaggle/working/mask_train_2014/"))/batch_size
model.fit(train_generator,validation_data = val_generator, steps_per_epoch = train_steps , epochs=20)

def print_preprocessed_image_shapes(model, generator):
    """
    Print the shapes of preprocessed images generated by the provided model and generator.

    Args:
        model (tf.keras.Model): The trained model.
        generator (CustomDataGenerator): Instance of the CustomDataGenerator class.
    """
    for i in range(len(generator)):
        # Get a batch of preprocessed images from the generator
        batch_images, batch_masks = generator[i]

        # Print the shapes of the preprocessed images
        for image in batch_images:
            print(f"Shape of preprocessed image: {image.shape}")
        for mask in batch_maskss:

             print(f"Shape of preprocessed image: {mask.shape}")
            
# Print the shapes of preprocessed images
print_preprocessed_image_shapes(model, val_generator)