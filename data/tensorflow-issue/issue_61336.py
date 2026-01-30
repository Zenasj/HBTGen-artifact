from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def create_model():
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
	model = models.Sequential()
	model.add(base_model)
	model.add(layers.GlobalAveragePooling2D())
	model.add(layers.Dense(10, activation='sigmoid'))
	return model

model = create_model()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('v5.keras'),
    monitor='val_f1_score',
    mode='max',
    save_best_only=True
)

model.compile(
	loss='binary_crossentropy',
optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
	metrics = [ tf.keras.metrics.F1Score(threshold=0.5)]
)

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=1,
    validation_data=(X_test, y_test),
    callbacks=[model_checkpoint]
)