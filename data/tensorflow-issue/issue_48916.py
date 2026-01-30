import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

class HourglasSequential(tf.keras.models.Model):
    def __init__(self):
        super(HourglasSequential, self).__init__()
        self.nrSH_in = 27 #number of input spherical harmonics coeff.
        self.baseFilter = 16
        self.nrSH_out = 9 #if gray else 27 #nr output SH
        self.ncPre = self.baseFilter #This is the amount required for the pre-convolution step

        self.ncHG3 = self.baseFilter #this is the amount of output channels for the first and last hourglass block
        self.ncHG2 = self.baseFilter * 2
        self.ncHG1 = self.baseFilter * 4
        self.ncHG0 = self.baseFilter * 8 + self.nrSH_in # Bottleneck layer. 
      
        self.pre_conv = SeparableConv2D(self.ncPre, kernel_size=(5,5), strides=(1,1), padding="same", name="pre_conv")
        self.pre_bn = BatchNormalization(name="pre_bn")
        self.relu_1 = ReLU()
        self.HG3_BB_Upper_conv1 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG3_BN1 = BatchNormalization()
        #self.HG3_upper_conv1 = SeparableConv2D(16, kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False) 
        #self.HG3_upper_conv2 = SeparableConv2D(16, kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False) 
        #self.HG3_upper_bn = BatchNormalization()
        #self.HG3_upper_relu_1 = ReLU()
        #self.HG3_upper_relu_2 = ReLU()
    def compute_output_shape(self, input_shape):
        return [tf.TensorShape((1,1,1,9))] #Must somehow add L_hat to the dimensions
    
    def setup_model(self):
        x = tf.keras.Input(shape=(512,512,1))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
    
    def call(self, inputs):
        x = inputs[0]
        feat = self.pre_conv(x)
        feat = self.pre_bn(feat)
        feat = self.relu_1(feat)
        #HG3 tar in  16 som ncout, ncin = 16 som IN
        #HG3 basic block upper som tar in 16 som input och 16 som output
        feat = self.HG3_BB_Upper_conv1(feat)
        feat = self.HG3_BN1(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(16,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG3 downsample
        feat = MaxPool2D(2,2)(feat)
        #HG3 LOW1 basic block med ncIn 16 och 16 Ncout
        feat = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(16,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG2 ncIn 16, ncout = 32
        #HG2 bb upper 
        feat = SeparableConv2D(16,3,1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(16,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG2 downsample
        feat = MaxPool2D(2,2)(feat)
        #HG2 LOW1 basic block med ncIn 16 och 16 Ncout
        feat = SeparableConv2D(32, 3, 1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(32,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG1 ncIn 32, ncout = 64
        #HG1 bb upper 
        feat = SeparableConv2D(32,3,1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(32,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG1 downsample
        feat = MaxPool2D(2,2)(feat)
        #HG1 LOW1 basic block med ncIn 16 och 16 Ncout
        feat = SeparableConv2D(64, 3, 1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(64,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG0 ncIn 64, ncout = 128 + 27 = 155
        #HG0 bb upper 
        feat = SeparableConv2D(64,3,1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(64,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #HG0 downsample
        feat = MaxPool2D(2,2)(feat)
        #HG0 LOW1 basic block
        feat = SeparableConv2D(155, 3, 1, padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        feat = SeparableConv2D(155,3,1,padding="same", use_bias=False)(feat)
        feat = BatchNormalization()(feat)
        feat = ReLU()(feat)
        #LightingNet ncIn 27, out 9, middle 128
        feat = feat[:,:,:,0:27]
        feat = tf.math.reduce_mean(feat, axis=(1,2), keepdims=True)
        feat = SeparableConv2D(128, 1, 1, use_bias=False)(feat)
        feat = PReLU()(feat)
        #print(feat.shape)
        L_hat = SeparableConv2D(9,1,1, use_bias=False)(feat)

        return L_hat