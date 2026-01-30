import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def run_train(dataset, num_epochs=2):
    start_time = time.perf_counter()

    model = VGGBase()

    for _ in tf.data.Dataset.range(num_epochs):
        for image,target in dataset: # (batch_size (N), 300, 300, 3)
            image = np.array(image)
            target = np.array(target)
            print(target)
            print(type(image), type(target),image.shape,target.shape)
            predicted_locs, predicted_socres = model(image)# (N, 8732, 4), (N, 8732, n_classes)
            print(predicted_locs,predicted_socres)
            pass
            break
        pass

def train():
    if isprint:print(tf.__version__)
    batch_size= 256

    #dataset test0
    images,boxes,labels,difficulties= PascalVOCDataset()
    boxes = tf.ragged.constant(boxes)
    dataset = tf.data.Dataset.from_tensor_slices((images,boxes))
    run_train(dataset.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE))

class  VGGBase(Model):
    def __init__(self):
        super(VGGBase,self).__init__()
        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # put this before your conv layer
        self.conv1_1 = tf.keras.layers.Conv2D(3, kernel_size=3,padding='same',strides=1, activation='relu'),
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same',strides=1,activation='relu'),
        self.pool1 = tf.keras.layers.MaxPool2D(2,2),

        self.conv2_1  =  tf.keras.layers.Conv2D(128, kernel_size=3, padding='same',strides= 1,activation='relu'),
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=3,padding='same',strides= 1,activation='relu'),
        self.pool2 = tf.keras.layers.MaxPool2D(2,2),

        self.conv3_1 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu'),
        self.conv3_2 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu'),
        self.conv3_3 =  tf.keras.layers.Conv2D(256, kernel_size=3, padding='same',strides= 1,activation='relu'),
        self.pool3 = tf.keras.layers.MaxPool2D(2,2),

        self.conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.pool4 = tf.keras.layers.MaxPool2D(2, 2),

        self.conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1,activation='relu'),
        self.pool5 = tf.keras.layers.MaxPool2D(2, 2),

        self.padding6 = tf.keras.layers.ZeroPadding2D(padding=(6, 6))  # put this before your conv layer
        self.conv6 = tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same',dilation_rate=6,activation='relu') # atrous convolution
        self.conv7 = tf.keras.layers.Conv2D(1024, kernel_size=1,activation='relu')
        #self.load_weights()
    def call(self,x):
        x = self.padding_1(x)
        x = self.conv1_1(x)# (N, 64, 300, 300)
        x = self.conv1_2(x)# (N, 64, 300, 300)
        x = self.pool1(x) # (N, 64, 150, 150)

        x = self.conv2_1(x) # (N, 128, 150, 150)
        x = self.conv2_2(x) # (N, 128, 150, 150)
        x = self.pool2(x)# (N, 128, 75, 75)

        x = self.conv3_1(x) # (N, 256, 75, 75)
        x = self.conv3_2(x)# (N, 256, 75, 75)
        x = self.conv3_3(x)# (N, 256, 75, 75)
        x = self.pool3(x) #(N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        x = self.conv4_1(x)# (N, 512, 38, 38)
        x = self.conv4_2(x)# (N, 512, 38, 38)
        x = self.conv4_3(x)# (N, 512, 38, 38)
        conv4_3_feats = x# (N, 512, 38, 38)
        x = self.pool4(x)# (N, 512, 19, 19)

        x = self.conv5_1(x) # (N, 512, 19, 19)
        x = self.conv5_2(x) # (N, 512, 19, 19)
        x = self.conv5_3(x) # (N, 512, 19, 19)
        x = self.pool5(x) # (N, 512, 19, 19), pool5 does not reduce dimensions

        x = self.padding6(x)
        x = self.conv6(x) # (N, 1024, 19, 19)
        x = self.conv7(x) # (N, 1024, 19, 19)
        conv7_feats = x

        return conv4_3_feats, conv7_feats