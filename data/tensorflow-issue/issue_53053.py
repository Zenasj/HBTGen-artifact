import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomContrast(.1),
        tf.keras.layers.RandomRotation(.5, fill_mode='nearest'),
    ]
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, channels))
base_model = tf.keras.applications.ResNet50V2(
    input_shape=(IMG_SIZE, IMG_SIZE, channels),
    include_top=False,
)
x = data_augmentation(inputs)
x = tf.keras.applications.resnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(dropout)(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        histogram_freq=5,
        write_graph=True,
        write_images=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_dir,
        monitor='val_accuracy',
        save_best_only=True,
    ),
]
model.fit(train_ds, epochs=100, callbacks=callbacks, validation_data=val_ds)