from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(380, 380, 3), include_preprocessing=True)

base_model.trainable = False

top_dropout_rate = 0.2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
x = BatchNormalization()(x)
x = Dropout(top_dropout_rate, name="top_dropout")(x)
x = Dense(1, activation='sigmoid', name="prediction")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=input_shape, include_preprocessing=True)
base_model.trainable = False

top_dropout_rate = 0.2
input_tensor = Input(shape=input_shape)
preprocessed_input = preprocess_input(input_tensor)

x = base_model(preprocessed_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
x = BatchNormalization()(x)
x = Dropout(top_dropout_rate, name="top_dropout")(x)
x = Dense(1, activation='sigmoid', name="prediction")(x)

model = Model(inputs=input_tensor, outputs=x)