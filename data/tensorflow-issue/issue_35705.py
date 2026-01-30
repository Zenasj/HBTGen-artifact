import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        "worker": ["10.10.1.168:1234"],
        'chief': ["10.10.1.60:2345"]
    },
    'task': {'type': 'chief', 'index': 0}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def get_label(file_path, class_names):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2] == class_names

def parse_image(filename):
    parts = tf.strings.split(filename, "\\")
    label = get_label(filename, CLASS_NAMES)
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [299,299])
    return image, label

def make_dataset_unbatched():
    images_ds = list_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    images_ds = images_ds.shuffle(BATCH_SIZE)
    images_ds = images_ds.repeat(epochs)
    images_ds = images_ds.prefetch(BUFFER_SIZE)
    return images_ds

datasetFilePath = "D:\TrainData\BalancedData"
IMAGESIZE = 299
AUTOTUNE = tf.data.experimental.AUTOTUNE
datasetPath = pathlib.Path(datasetFilePath)
list_ds = tf.data.Dataset.list_files(str(datasetPath/"*/*"))
num_elements = tf.data.experimental.cardinality(list_ds).numpy()

CLASS_NAMES = np.array([item.name for item in datasetPath.glob('*')])

epochs = 2
def build_and_compile_model():
    base_model =tf.keras.applications.InceptionV3(include_top=False, weights = "imagenet", input_shape=(299,299,3))

    base_model.trainable = True
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    predictions = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    base_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])
    return model
logdir = os.path.join("Z:\Tensorflow\TensorboardLogs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="Z:\Tensorflow\Checkpoints"), 
             tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)]

with strategy.scope():
    dataset = make_dataset_unbatched().batch(BATCH_SIZE, drop_remainder=True)
    multi_worker_model = build_and_compile_model()

history = multi_worker_model.fit(dataset, epochs=epochs, steps_per_epoch=50, callbacks=callbacks)