import tensorflow as tf
from tensorflow import keras

# Assuming we use mnist data set
model = Sequential([
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, callbacks=[tf.keras.callbacks.ProgbarLogger('steps')])