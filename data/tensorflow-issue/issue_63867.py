import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

def main():

    # Removed dependance on argsparser()
    # args = argsparer()

    # batch_size = args.batchSize # 32
    batch_size = 32
    # img_height = args.height    # 128
    img_height = 128
    # img_width = args.width      # 128
    img_width = 128
    activation1 = 'swish'
    activation2 = 'elu'
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)
    # epochs = args.epochs        # 1
    epochs = 1

    train_data_dir = pathlib.Path("{}/{}/".format(args.inputDir, args.trainingSubDir)).with_suffix('')
    val_data_dir = pathlib.Path("birds/birds/valid/".format(args.inputDir, args.validationSubDir)).with_suffix('')

    # assume all files in training directory are readable image files.
    image_count = len(list(train_data_dir.glob('*/*.*')))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        seed=random.randint(1, 10000),
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_data_dir,
        # validation_split=0.1,
        # subset="validation",
        seed=random.randint(1, 10000),
        image_size=(img_height, img_width),
        batch_size=batch_size)

    num_classes = len(train_ds.class_names)
    print("Training directory '{}' contains {} images in {} categories.".format(train_data_dir, image_count, num_classes))

    normalization_layer = layers.Rescaling(1. / 255)
    train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=False)
    val_ds_norm = val_ds.map(lambda x, y: (normalization_layer(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=False)

    # train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        # layers.Conv2D(16, 3, padding='same', activation=activation1),
        # layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=activation1),
        layers.MaxPooling2D(),
        layers.Conv2D(48, 3, padding='same', activation=activation2),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=activation1),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),
        layers.Flatten(),
        layers.Dense(128, activation=activation2),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(
        train_ds_norm,
        validation_data=val_ds_norm,
        epochs=epochs)


    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)