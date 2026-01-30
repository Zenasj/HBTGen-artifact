import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

BATCH_SIZE = 100
IMG_SHAPE  = 150

image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE,IMG_SHAPE))
val_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=val_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE,IMG_SHAPE))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
epochs = 100
model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen, 
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.data_adapter import ListsOfScalarsDataAdapter
from tensorflow.python.keras.engine.data_adapter import TensorLikeDataAdapter
from tensorflow.python.keras.engine.data_adapter import GenericArrayLikeDataAdapter
from tensorflow.python.keras.engine.data_adapter import DatasetAdapter
from tensorflow.python.keras.engine.data_adapter import GeneratorDataAdapter
from tensorflow.python.keras.engine.data_adapter import CompositeTensorDataAdapter

data_adapter.ALL_ADAPTER_CLS = [
 ListsOfScalarsDataAdapter,
 TensorLikeDataAdapter,
 GenericArrayLikeDataAdapter,
 DatasetAdapter,
 GeneratorDataAdapter,
#  tensorflow.python.keras.engine.data_adapter.KerasSequenceAdapter,
 CompositeTensorDataAdapter      
]

data_adapter.ALL_ADAPTER_CLS