import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

res=tf.keras.applications.ResNet50()
grad_model = tf.keras.models.Model(
    [res.inputs], [res.layers[-5].output, res.output]
)
grad_model.summary()

class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(64)
        self.out = Dense(10, activation='softmax')

    def call(self, x):
        if(len(x.shape)==3):
            x=tf.expand_dims(x,axis=0)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
lenet=LeNet()
inputs=tf.keras.layers.Input((28,28,1))
lenet(inputs)
lenet.summary()
grad_model = tf.keras.models.Model(
    [lenet.inputs], [lenet.layers[-5].output, lenet.output]
)
grad_model.summary()