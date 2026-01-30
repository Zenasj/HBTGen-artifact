from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    import tensorflow as tf
    import matplotlib.pyplot as plt

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # This pyplot block of code contributes to the error. Without it, model.train functions fine.
    # Begin Block
    def display_image(position):
        image = x_train[position].squeeze()
        plt.title('Example %d. Label: %d' % (position, y_train[position]))
        plt.imshow(image, cmap='gray')
    display_image(0)
    # End block

    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    """
    # This will throw the error
    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
    """
    model.fit(x_train, y_train, epochs=5, verbose=1)

    # test accuracy
    model.evaluate(x_test, y_test)