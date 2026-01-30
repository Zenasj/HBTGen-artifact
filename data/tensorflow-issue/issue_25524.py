def inception(input_shape: Tuple[int, int, int]) -> k.Model:
    # set the input
    input_img = k.layers.Input(shape=input_shape)

    tower_1 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")(input_img)
    tower_1 = k.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(tower_1)

    tower_2 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")(input_img)
    tower_2 = k.layers.Conv2D(64, (5, 5), padding="same", activation="relu")(tower_2)

    tower_3 = k.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(input_img)
    tower_3 = k.layers.Conv2D(64, (1, 1), padding="same", activation="relu")(tower_3)

    output = k.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    output = k.layers.Flatten()(output)
    out = k.layers.Dense(10, activation="softmax")(output)

    model = k.models.Model(inputs=[input_img], outputs=[out])

    print(model.summary())

    return model

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras as k

import multiprocessing
import model

DATADIR = "/run/media/federico/XData/tensorflow_datasets"
CLASSES = 10
EPOCHS = 25
BATCH = 128
LRATE = 0.01
DECAY = LRATE / EPOCHS

# this is applied to every single data passed
def process_features(feature):
    image, label = feature["image"], feature["label"]

    # image conversion into [0, 1]
    image = image / 255
    image = tf.cast(image, tf.float32)
    feature["image"] = image

    # label conversion into one-hot
    print(label)
    label = tf.one_hot(label, CLASSES)
    feature["label"] = label

    return feature

# print(tfds.list_builders())

dataset, dataset_info = tfds.load(name="cifar10", data_dir=DATADIR, with_info=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

# process features
train_dataset = (
    train_dataset.map(process_features, num_parallel_calls=multiprocessing.cpu_count())
    .shuffle(1000)
    .batch(BATCH)
    .repeat(EPOCHS)
    # .prefetch(10)
)


test_dataset = test_dataset.map(
    process_features, num_parallel_calls=multiprocessing.cpu_count()
).batch(dataset_info.splits["test"].get_proto().statistics.num_examples)

for f in test_dataset:
    images_t, labels_t = f["image"], f["label"]

model = model.inception(dataset_info.features["image"].shape)

# compile the model
model.compile(
    optimizer=k.optimizers.RMSprop(learning_rate=LRATE, decay=DECAY),
    loss=k.losses.categorical_crossentropy,
    metrics=[k.metrics.categorical_accuracy],
)

tbCallback = [
    k.callbacks.TensorBoard(
        log_dir="./log", histogram_freq=0, write_graph=False, write_images=False
    )
]
# tbCallback = [k.callbacks.TensorBoard(log_dir='./log')]

step = 0
for f in train_dataset:
    images, labels = f["image"], f["label"]
    step += 1
    model.fit(
        images,
        labels,
        validation_data=(images_t, labels_t),
        steps_per_epoch=100,
        epochs=1,
        batch_size=BATCH,
        callbacks=tbCallback,
    )