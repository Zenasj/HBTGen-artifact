import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

from keras.datasets import fashion_mnist

tf.random.set_seed(1)
np.random.seed(1)


if __name__ == '__main__':
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255
    test_X = test_X / 255

    train_L = len(train_X)
    test_L = len(test_X)

    print(train_Y[0], type(train_Y[0]))

    x_input = tf.keras.Input(dtype=tf.dtypes.float32, shape=[28, 28, 1])
    y_input = tf.keras.Input(dtype=tf.dtypes.int32, shape=[1])
    
    x = tf.keras.layers.Flatten()(x_input)
    y_target = tf.one_hot(y_input, 10)
    y_logits = tf.keras.layers.Dense(10)(x)
    y_probability = tf.keras.layers.Softmax()(y_logits)
    y_output = tf.math.argmax(
        input=y_probability,
        axis=1,
        output_type=tf.dtypes.int32
    )

    model = tf.keras.Model(inputs=[x_input, y_input], outputs=y_output)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),

        metrics=[
            # 'accuracy'
            #tf.keras.metrics.Accuracy()
        ]
    )

    # Trains for 5 epochs
    model.fit(x=train_X, y=train_Y, batch_size=32, epochs=5)

import numpy as np
import tensorflow as tf

from mnist import train_images, test_images, train_labels, test_labels

tf.random.set_seed(1)
np.random.seed(1)

LOSS_TYPES = {
    'CE': tf.keras.losses.CategoricalCrossentropy(),
    'SCE': tf.keras.losses.SparseCategoricalCrossentropy()
}
LOSS_TYPE1 = 'CE'
LOSS_TYPE2 = LOSS_TYPES[LOSS_TYPE1]

DTYPES = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64
}

DTYPE1 = 'float32'
DTYPE2 = DTYPES[DTYPE1]

tf.keras.backend.set_floatx(DTYPE1)

EPOCHS = 50
NCLASS = 10
BATCH_SIZE = 256


class MNISTModel(tf.keras.Model):
    loss_object = LOSS_TYPE2

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validate_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    validate_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    n_class = None

    def __init__(self, n_class):
        super(MNISTModel, self).__init__()
        self.n_class = n_class

        self.d0 = tf.keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu')
        self.d1 = tf.keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu')
        self.d2 = tf.keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu')
        self.d3 = tf.keras.layers.Conv2D(16, [3, 3], 1, 'same', activation='relu')

        self.p0 = tf.keras.layers.MaxPooling2D((2, 2))
        self.p1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.p2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.p3 = tf.keras.layers.MaxPooling2D((2, 2))

        self.d_prior = tf.keras.layers.Dense(64, activation='relu')
        self.d_out = tf.keras.layers.Dense(self.n_class, activation='softmax')

    @tf.function
    def call(self, x, training):
        x = tf.dtypes.cast(
            tf.reshape(x, [-1, 28, 28, 1]),
            dtype=DTYPE2
        )

        x = self.d0(x)
        x = self.p0(x)

        x = self.d1(x)
        x = self.p1(x)

        x = self.d2(x)
        x = self.p2(x)

        x = self.d3(x)
        x = self.p3(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.d_prior(x)
        x = self.d_out(x)
        return x

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.call(x, training=True)

            if LOSS_TYPE1 == 'CE':
                y_one_hot = tf.one_hot(y, self.n_class)
                loss = self.loss_object(y_one_hot, predictions)
            elif LOSS_TYPE1 == 'SCE':
                loss = self.loss_object(y, predictions)

            # print('loss', loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        # print([gvs.shape for gvs in gradients])

        # gradients = [
        #     tf.where(
        #         tf.math.is_nan(grad),
        #         tf.zeros_like(grad),
        #         grad
        #     )
        #     for grad in gradients
        # ]

        # print('gradients', gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_acc.update_state(y, predictions)
        return loss

    @tf.function
    def test_step(self, x, y):
        predictions = self.call(x, training=False)
        if LOSS_TYPE1 == 'CE':
            y_one_hot = tf.one_hot(y, self.n_class)
            t_loss = self.loss_object(y_one_hot, predictions)
        elif LOSS_TYPE1 == 'SCE':
            t_loss = self.loss_object(y, predictions)
        # t_loss = tf.reduce_sum(t_loss) * (1. / BATCH_SIZE)
        self.test_loss(t_loss)
        self.test_acc.update_state(y, predictions)

    @tf.function
    def validate_step(self, x, y):
        predictions = self.call(x, training=False)
        if LOSS_TYPE1 == 'CE':
            y_one_hot = tf.one_hot(y, self.n_class)
            v_loss = self.loss_object(y_one_hot, predictions)
        elif LOSS_TYPE1 == 'CE':
            v_loss = self.loss_object(y, predictions)
        # t_loss = tf.reduce_sum(t_loss) * (1. / BATCH_SIZE)
        self.validate_loss(v_loss)
        self.validate_acc.update_state(y, predictions)


if __name__ == '__main__':
    x_train = train_images()
    y_train = train_labels()

    x_test = test_images()
    y_test = test_labels()

    print(np.shape(y_train))

    model = MNISTModel(NCLASS)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        for x, y in train_ds:
            model.train_step(x, y)

        for x, y in test_ds:
            model.test_step(x, y)

        template = 'Epoch {}, Train Loss: {}, Test Loss: {}, Train Acc: {}, Test Acc: {}'

        epoch_str = template.format(
            epoch + 1,
            model.train_loss.result(),
            model.test_loss.result(),
            model.train_acc.result(),
            model.test_acc.result()
        )
        print(epoch_str)

        # Reset the metrics for the next epoch
        model.train_loss.reset_states()
        model.test_loss.reset_states()
        model.train_acc.reset_states()
        model.test_acc.reset_states()

y_pred = model.predict(x_train)
predictions = tf.argmax(y_pred,axis=1)