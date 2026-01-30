from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
cfg_imagenet = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
cfg_cifar = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

class MobileNet(tf.keras.Model):
    def __init__(self, data_format='channels_last'):
        super(MobileNet, self).__init__()
        self.channel_axis = 1 if data_format == 'channels_last' else -1
        self.initializer = tf.keras.initializers.he_normal()

        self.features = self._make_layers()
        self.classifier = tf.keras.layers.Dense(10, kernel_initializer='he_normal')

    def _make_layers(self):
        layers = []
        layers.append(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', name='conv1',
                      kernel_initializer=self.initializer))
        layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name='conv1_bn'))
        layers.append(tf.keras.layers.ReLU(name='conv1_relu'))
        for idx, x in enumerate(cfg_imagenet):
            i = idx + 1
            filters = x if isinstance(x, int) else x[0]
            strides = 1 if isinstance(x, int) else x[1]
            layers.append(tf.keras.layers.DepthwiseConv2D((3, 3), strides, padding='same', kernel_initializer=self.initializer, name='block{}_dw'.format(i)))
            layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name='block{}_bn1'.format(i)))
            layers.append((tf.keras.layers.ReLU(name='block{}_relu1'.format(i))))
            layers.append(tf.keras.layers.Conv2D(filters, 1, padding='same', activation='relu', kernel_initializer=self.initializer, name='block{}_pw'.format(i)))
            layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name='block{}_bn2'.format(i)))
            layers.append((tf.keras.layers.ReLU(name='block{}_relu2'.format(i))))
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        return layers

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def loss_func(model, x, y):
    y_ = model(x)
    return criterion(y, y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
for epoch in range(10):
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        train_acc(y, model(x))
        print(loss_value)
        print(train_acc.result())
    print(epoch, "complete")