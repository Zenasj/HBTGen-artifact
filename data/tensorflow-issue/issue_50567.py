from tensorflow import keras
from tensorflow.keras import losses,layers,optimizers,Model
import tensorflow as tf
import numpy as np 

class Test(layers.Layer):
    def __init__(self):
        super().__init__() 
        self.f = self.add_weight(name = 'kernel', 
                                 trainable = True,
                                 shape = [12,12,3,3], 
                                 initializer = tf.random_uniform_initializer()) 
    def call(self, inp):  
        print(inp.shape)
        _,H,W,C = inp.shape
        y = tf.nn.conv_transpose(inp, self.f, [-1,H*4, W*4,C], (4,4), 'SAME') 
        return y 

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu) 
 
with strategy.scope():
    model = keras.Sequential()
    model.add(Test())
    model.build([1,48, 48, 3])
    model.compile(optimizers.Adam(), losses.mean_absolute_error)

lr_data = np.zeros([256, 48, 48, 3]).astype(np.float32)
hr_data = np.zeros([256, 48*4, 48*4, 3]).astype(np.float32)
print('Fit Begin')
model.fit(lr_data, hr_data, epochs=5, batch_size=32, verbose=1)  
print('Fit End')