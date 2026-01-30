from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D,GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import L2
class ResNetBlock(Model):
    def __init__(self, filters=64, strides=1):
        super(ResNetBlock, self).__init__()
        self.strides = strides
        self.c1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()

        if(strides > 1):
            self.c3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
            self.b3 = BatchNormalization()
        
        self.a2 = Activation('relu')

    def call(self, inputs):
        short_x = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x)
        if(self.strides > 1):
            short_x = self.c3(short_x)
            short_x = self.b3(short_x)
        return self.a2(short_x + y)
    
class ResNet(Model):
    def __init__(self, model_lst, cur_filters = 64):
        super(ResNet, self).__init__()
        self.c1 = Conv2D(filters=cur_filters, kernel_size=(7, 7), strides=2, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D((2, 2), 2)
        self.blocks = Sequential()
        for (i, lst) in enumerate(model_lst):
            for ids in range(lst):
                if(i != 0 and ids == 0):
                    block = ResNetBlock(cur_filters, strides=2)
                else:
                    block = ResNetBlock(cur_filters, strides=1)
                self.blocks.add(block)    
            cur_filters *= 2
        self.g1 = GlobalAveragePooling2D()
        self.d1 = Dense(10, activation='softmax', kernel_regularizer=L2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.blocks(x)
        x = self.g1(x)
        y = self.d1(x)
        return y


# ---------------------------------------------
# ResNet18
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.family']=['SimHei', 'Arial']
from tensorflow.keras import *
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean,SparseCategoricalAccuracy  
from tensorflow.keras.datasets.fashion_mnist import load_data 
batch_size = 64
epochs = 20
validation_freq = 2
(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train/255., x_test/255.
x_train = np.expand_dims(x_train, -1).astype(np.float32)
x_test = np.expand_dims(x_test, -1).astype(np.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(len(x_test)).batch(batch_size)

model = ResNet([2, 2, 2, 2])
losses = SparseCategoricalCrossentropy(from_logits=False)
optimizer = Adam()
train_metrics_loss = Mean()
train_metrics_accuracy = SparseCategoricalAccuracy()
test_metrics_loss = Mean()
test_metrics_accuracy = SparseCategoricalAccuracy()

train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []
@tf.function
def train_step(model, input_images, y_real):
    with tf.GradientTape() as tape:
        y_pred = model(input_images, training=True)
        loss = losses(y_real, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metrics_loss.update_state(loss)
    train_metrics_accuracy.update_state(y_real, y_pred)
@tf.function
def test_step(model, input_images, y_real):
    with tf.GradientTape() as tape:
        y_pred = model(input_images, training=False)
        loss = losses(y_real, y_pred)
    test_metrics_loss.update_state(loss)
    test_metrics_accuracy.update_state(y_real, y_pred)

for epoch in range(epochs):
    train_metrics_loss.reset_states()
    train_metrics_accuracy.reset_states()
    test_metrics_accuracy.reset_states()
    test_metrics_loss.reset_states()
    for x_batch, y_batch in train_dataset:
        train_step(model, x_batch, y_batch)
    train_losses.append(train_metrics_loss.result())
    train_accuracy.append(train_metrics_accuracy.result())
    print(f"epoch={epoch}, train_loss={train_metrics_loss.result()}, train_accuracy={train_metrics_accuracy.result()}")
    if(epoch % validation_freq == 0):
        for test_x_batch, test_y_batch in test_dataset:
            test_step(model, test_x_batch, test_y_batch)
        test_losses.append(test_metrics_loss.result())
        test_accuracy.append(test_metrics_accuracy.result())
        print(f"epoch={epoch}, test_loss={test_metrics_loss.result()}, test_accuracy={test_metrics_accuracy.result()}")



plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.title('损失值变化图')
plt.plot(test_losses, 'g-', label="Test_Loss")
plt.plot(train_losses, 'r-', label="Train_Loss")

plt.legend()

plt.subplot(1, 2, 2)
plt.title("准确率变化图")
plt.plot(train_accuracy, 'r-', label="Train_Accuracy")
plt.plot(test_accuracy, 'g-', label="Test_Accuracy")
plt.legend()

plt.show()