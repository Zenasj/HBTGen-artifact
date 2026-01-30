from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

# Mnist dataset
IMAGE_ROW, IMAGE_COLS = 28, 28
NUM_CLASSES = 10
BATCH_SIZE = 32

temp_dir = './temp'


def get_input_datasets(use_bfloat16=False):
    """Creates train and test dataset objects for mnist dataset.
    Args:
      use_bfloat16: Boolean, to determine if input should be cast to bfloat16
    Returns:
      Train dataset, test dataset and input shape, and class names.
    """

    cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, IMAGE_ROW, IMAGE_COLS)
        x_test = x_test.reshape(x_test.shape[0], 1, IMAGE_ROW, IMAGE_COLS)
        input_shape = (1, IMAGE_ROW, IMAGE_COLS)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMAGE_ROW, IMAGE_COLS, 1)
        x_test = x_test.reshape(x_test.shape[0], IMAGE_ROW, IMAGE_COLS, 1)
        input_shape = (IMAGE_ROW, IMAGE_COLS, 1)

    # Preprocess
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # build dataset
    # ds_train = tf.data.Dataset.from_tensor_slices({'images': x_train, 'labels': y_train})
    # ds_test = tf.data.Dataset.from_tensor_slices({'images': x_test, 'labels': y_test})

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))

    # Preprocess dataset
    ds_train = preprocess_dataset(ds_train, cast_dtype)
    ds_valid = preprocess_dataset(ds_valid, cast_dtype)
    ds_test = preprocess_dataset(ds_test, cast_dtype)

    return ds_train, ds_valid, ds_test, input_shape, class_names


def preprocess_dataset(dataset, cast_dtype):
    dataset = dataset.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    dataset = dataset.shuffle(buffer_size=6000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset


def plot_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)


def test_ds(dataset, fname):
    save_dir = os.path.join(temp_dir, "%s.png" % fname)
    os.makedirs(temp_dir, exist_ok=True)
    plt.figure()
    for image, label in dataset.take(1):
        for index in range(4):
            plt.subplot(2, 2, index + 1)
            plt.imshow(image[index].numpy().reshape(28, 28))
            plt.xlabel(label[index].numpy())
            plt.grid(False)
    plt.savefig(save_dir, bbox_inches="tight")
    plt.clf()

def get_optimizer(optimizer_choice='SGD', learning_rate=0.01, momentum=0.9):
    return {
        'SGD':tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum),
        'Adam': tf.keras.optimizers.Adam(lr=learning_rate)
    }.get(optimizer_choice, 'SGD')
    #
    # return {
    #     'SGD':tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum),
    #     'Adam': tf.train.AdamOptimizer(learning_rate=learning_rate)
    # }.get(optimizer_choice, 'SGD')


def create_model(input_shapes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shapes))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    return model


def create_model_functional(input_shapes, kernel_size=(3, 3), dropout_rate=0, l2_regularizer=0.1):
    input_tensor = tf.keras.Input(shape=input_shapes)
    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(input_tensor)
    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(layer)
    layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = tf.keras.layers.Dropout(rate=dropout_rate)(layer)
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(rate=dropout_rate)(layer)
    predictions = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularizer))(layer)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=predictions)
    return model

def fit_model_and_evaluate(model, optimizer_choice='SGD', learning_rate=0.01, verbose=1):
    optimizer = get_optimizer(optimizer_choice=optimizer_choice, learning_rate=learning_rate)
    os.makedirs('graph', exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    file_path = 'checkpoint/model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path)
    # log_dir = os.path.join()
    board = tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
                                           write_graph=True, write_images=True)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.fit(x=ds_train, validation_data=ds_valid, epochs=20, steps_per_epoch=468,
              callbacks=[board, model_checkpoint],
              validation_steps=40, verbose=verbose)
    score = model.evaluate(ds_test, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    ds_train, ds_valid, ds_test, input_shapes, class_names = get_input_datasets()
    model = create_model_functional(input_shapes)
    fit_model_and_evaluate(model)
    # test_ds(ds_train, 'train')
    # test_ds(ds_test, 'test')

model.fit(x=ds_train, validation_data=ds_valid, epochs=20, steps_per_epoch=468,
              callbacks=[board, model_checkpoint],
              validation_steps=40, verbose=verbose)

model.fit(x=ds_train, epochs=20, steps_per_epoch=468,
              callbacks=[board, model_checkpoint], verbose=verbose)