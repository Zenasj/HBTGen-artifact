from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def process_data(img, lbl):
    img = tf.image.resize(img, (96, 96))
    img = (img-128) / 128
    return img, lbl
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
train_data = train_data.map(process_data)
test_data = test_data.map(process_data)
train_data, test_data

# load the pretrained model
base_model = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, pooling='avg')
x = base_model.outputs[0]
outputs = layers.Dense(10, activation=tf.nn.softmax)(x)
model = keras.Model(inputs=base_model.inputs, outputs=outputs)

# Trained with keras fit
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_data, epochs=1)

# The results are: loss: 0.4345 - accuracy: 0.8585

# Trained with tf.GradientTape
optimizer = keras.optimizers.Adam()
train_loss = keras.metrics.Mean()
train_acc = keras.metrics.SparseCategoricalAccuracy()
def train_step(data, labels):    
    with tf.GradientTape() as gt:
        pred = model(data)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels, pred)

    grads = gt.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, pred)

model = keras.Model(inputs=base_model.inputs, outputs=outputs)
for xs, ys in train_data:
    train_step(xs, ys)

print('train_loss = {:.3f}, train_acc = {:.3f}'.format(train_loss.result(), train_acc.result()))

# The results are:  train_loss = 12.832, train_acc = 0.099