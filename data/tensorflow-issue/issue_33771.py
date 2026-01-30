from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
import glob
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, Conv2DTranspose, add, concatenate
from scipy.io import loadmat
import numpy as np
# from mobilev3 import MobileNetV3Large
from vgg_pr import VGG_PR
from tensorflow.keras.callbacks import TensorBoard
import logging
import cv2

# parameters
img_size = (299,299)
batch_size = 8
num_label = 20
initial_lr = 0.001
total_epoch = 100
repeat_times = 5
# case number
case_num = 9

os.chdir(os.getcwd())

train_img_list = sorted(glob.glob('../dataset/train/intensity/*.mat'))
train_label_list = sorted(glob.glob('../dataset/train/phase/*.txt'))
val_img_list = sorted(glob.glob('../dataset/validate/intensity/*.mat'))
val_label_list = sorted(glob.glob('../dataset/validate/phase/*.txt'))
ckpt_path = '../checkpoints/VGG-{epoch}.ckpt'
log_path = '../log/{}/'
if not os.path.exists(log_path.format(case_num)):
    os.mkdir(log_path.format(case_num))

# read data
def read_img(filename):
    image_dict = loadmat(filename.decode('utf-8'))
    exp_thresh = 1e4
    image_decoded = image_dict['Iz']
    image_decoded = cv2.resize(image_decoded, img_size, interpolation=cv2.INTER_AREA)
    image_decoded[image_decoded>exp_thresh] = exp_thresh
    image_decoded /= exp_thresh
    image_resized = np.float32(np.expand_dims(image_decoded, axis=-1))
    return image_resized

def read_label(filename):
    label = open(filename).read()
    label = label.strip().split(' ')
    label = [np.float32(i) for i in label if i!='']
    label = np.reshape(label, [1,1,-1])
    label = np.array(label) + 0.5  
    return label
def parse_function(image_filename, label_filename):
    img = tf.numpy_function(read_img, [image_filename], tf.float32)
    label = tf.numpy_function(read_label, [label_filename], tf.float32)
    return img, label
def train():
    logging.basicConfig(level=logging.INFO)
    tdataset = tf.data.Dataset.from_tensor_slices((train_img_list[:200], train_label_list[:200]))
    tdataset = tdataset.map(parse_function, 3).shuffle(buffer_size=200).batch(batch_size).repeat(repeat_times)
    vdataset = tf.data.Dataset.from_tensor_slices((val_img_list[:100], val_label_list[:100]))
    vdataset = vdataset.map(parse_function, 3).batch(batch_size)

    ### Vgg model
    model = VGG_PR(num_classes=num_label)

    logging.info('Model loaded')

    start_epoch = 0
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('training from scratch since weights no there')

    ######## training loop ########
    loss_object = tf.keras.losses.MeanSquaredError()
    val_loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    train_loss = tf.metrics.Mean(name='train_loss') 
    val_loss = tf.metrics.Mean(name='val_loss')
    writer = tf.summary.create_file_writer(log_path.format(case_num))

    with writer.as_default():
        for epoch in range(start_epoch, total_epoch):
            print('start training')
            try:
                for batch, data in enumerate(tdataset):
                    images, labels = data
                    with tf.GradientTape() as tape:
                        pred = model(images, training=True)
                        if len(pred.shape) == 2:
                            pred = tf.reshape(pred,[-1, 1, 1, num_label])
                        loss = loss_object(pred, labels)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    if batch % 20 ==0:
                        logging.info('Epoch: {}, iter: {}, loss:{}'.format(epoch, batch, loss.numpy()))
                    tf.summary.scalar('train_loss', loss.numpy(), step=epoch*1250*repeat_times+batch)      # the tdataset has been repeated 5 times..
                    tf.summary.text('Zernike_coe_pred', tf.as_string(tf.squeeze(pred)), step=epoch*1250*repeat_times+batch)
                    tf.summary.text('Zernike_coe_gt', tf.as_string(tf.squeeze(labels)), step=epoch*1250*repeat_times+batch)

                    writer.flush()
                    train_loss(loss)
                model.save_weights(ckpt_path.format(epoch=epoch))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)
            # validation step
            for batch, data in enumerate(vdataset):
                images, labels = data
                val_pred = model(images, training=False)
                if len(val_pred.shape) == 2:
                    val_pred = tf.reshape(val_pred,[-1, 1, 1, num_label])
                v_loss = val_loss_object(val_pred, labels)
                val_loss(v_loss)
            logging.info('Epoch: {}, average train_loss:{}, val_loss: {}'.format(epoch, train_loss.result(), val_loss.result()))
            tf.summary.scalar('val_loss', val_loss.result(), step = epoch)
            writer.flush()
            train_loss.reset_states()
            val_loss.reset_states()
        model.save_weights(ckpt_path.format(epoch=epoch))

import tensorflow as tf

# ------------------------------- Layers part -------------------------------
class BatchNormalization(tf.keras.layers.Layer):
    """All our convolutional layers use batch-normalization
    layers with average decay of 0.99.
    """

    def __init__(self):
        super().__init__(name="BatchNormalization")
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.99,
            name="BatchNorm")

    def call(self, input, training):
        return self.bn(input, training)

class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
            self,
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            padding='same',
            name='conv'):
        super().__init__(name="ConvBnAct")

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name)
        # self.norm = BatchNormalization()
        self.norm = tf.keras.layers.BatchNormalization(name='BatchNorm')

    def call(self, input, training):
        x = self.conv(input)
        x = self.norm(x,training=training)
        return x

class Block_1(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_1")
        self.conv1 = ConvBnAct(64,name='block1_conv1')
        self.conv2 = ConvBnAct(64,name='block1_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_2(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_2")
        self.conv1 = ConvBnAct(128,name='block2_conv1')
        self.conv2 = ConvBnAct(128,name='block2_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.pool(x)
        return x

class Block_3(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_3")
        self.conv1 = ConvBnAct(256,name='block3_conv1')
        self.conv2 = ConvBnAct(256,name='block3_conv2')
        self.conv3 = ConvBnAct(256,name='block3_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    def call(self, input ,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_4(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_4")
        self.conv1 = ConvBnAct(512,name='block4_conv1')
        self.conv2 = ConvBnAct(512,name='block4_conv2')
        self.conv3 = ConvBnAct(512,name='block4_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class Block_5(tf.keras.layers.Layer):
    def __init__(
            self):
        super().__init__(name="Block_5")
        self.conv1 = ConvBnAct(512,name='block5_conv1')
        self.conv2 = ConvBnAct(512,name='block5_conv2')
        self.conv3 = ConvBnAct(512,name='block5_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    def call(self, input,training=False):
        x = self.conv1(input,training)
        x = self.conv2(x,training)
        x = self.conv3(x,training)
        x = self.pool(x)
        return x

class VGG_PR(tf.keras.Model):
    def __init__(self,num_classes):
        super(VGG_PR, self).__init__()
        self.block1 = Block_1()
        self.block2 = Block_2()
        self.block3 = Block_3()
        self.block4 = Block_4()
        self.block5 = Block_5()
        self.avg = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', name='fc1')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', name='fc2')
        self.fc3 = tf.keras.layers.Dense(num_classes,activation='linear',name='predictions')

    def call(self, input, training=False):
        x = self.block1(input,training)
        x = self.block2(x,training)
        x = self.block3(x,training)
        x = self.block4(x,training)
        x = self.block5(x,training)
        x = self.avg(x)
        # print("output1:{}".format(x))
        x = self.fc1(x)
        # print("output2:{}".format(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x