from tensorflow.keras import layers
from tensorflow.keras import models

model = ...  # Get model (Sequential, Functional Model, or Model subclass)
model.save('path/to/location')

from tensorflow import keras
model = keras.models.load_model('path/to/location')

base_model = tf.keras.applications.ConvNeXtBase(
    include_top=False,  
    weights='imagenet',  
    input_shape=(image_height, image_width, 3)  
)
base_model.trainable = False

classification_head = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

transfer_model = tf.keras.Sequential([
    base_model,
    classification_head
])

model_filename = os.path.join(save_folder, f"{model_save_name}_TL.h5")
transfer_model.save(model_filename)

model_filename = os.path.join(save_folder, f"{model_save_name}_TL.keras")
transfer_model.save(model_filename)

model_filename = os.path.join(save_folder, f"{model_save_name}_TL_tf")
transfer_model.save(model_filename,save_format="tf")

import tensorflow as tf
from tensorflow import keras
from keras.applications.convnext import ConvNeXtBase, preprocess_input
import os

print(tf.__version__)

model_save_name ="OG_ConvNeXTBase_V2"
save_folder = "/content/mydata/Models/Test_Space"

test_model = tf.keras.applications.ConvNeXtBase(
    model_name="convnext_base",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


model_filename = os.path.join(save_folder, f"{model_save_name}_TL.h5")
test_model.save(model_filename)

model_filename = os.path.join(save_folder, f"{model_save_name}_TL.keras")
test_model.save(model_filename)

load_model_filename = '/content/mydata/Models/Test_Space/OG_ConvNeXTBase_V2_TL.h5'

load_test = tf.keras.models.load_model(load_model_filename)

load_model_filename = '/content/mydata/Models/Test_Space/OG_ConvNeXTBase_V2_TL.keras'

load_test = tf.keras.models.load_model(load_model_filename)