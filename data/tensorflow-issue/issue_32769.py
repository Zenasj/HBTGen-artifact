import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(len(label_names), activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['categorical_accuracy'])

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weight_only=True)
]

epochs = 10
history = model.fit(train_dataset, 
                    epochs=epochs,
                    callbacks=callbacks)