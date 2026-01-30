import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

inp_arr_without_ds = np.random.rand(2, 200, 200, 3)

inp_arr_4_ds = np.random.rand(2, 2, 200, 200, 3)
tf_ds = tf.data.Dataset.from_tensor_slices((inp_arr_4_ds, inp_arr_4_ds))
tf_ds = tf_ds.map(lambda x, y: (x, y)).repeat(10).shuffle(10)

class random_model(tf.keras.Model):
    
    def __init__(self, name):
        super(random_model, self).__init__(name=name)
        self.conv_1 = tf.keras.layers.Conv2D(3, [3, 3], padding="same")
        self.tf_board_writer = tf.summary.create_file_writer("test")
        self.img_callback = [tf.keras.callbacks.LambdaCallback(on_epoch_end=self.save_img)]
        
    def call(self, inputs):
        self.initialized_layer = self.conv_1(inputs)
        return self.initialized_layer
    
    def save_img(self, epochs, logs):
        with self.tf_board_writer.as_default():
            # Does not work: type: class 'tensorflow.python.framework.ops.Tensor' 
            tf.summary.image("image", self.initialized_layer, step=epochs)
            
            # Does work type: class 'tensorflow.python.framework.ops.EagerTensor'
            #tf.summary.image("image", tf.random.uniform([2, 200, 200, 3]), step=epochs) 
            
    def compile_model(self):
        self.compile(tf.optimizers.Adam(0.001), tf.losses.mean_absolute_error)
    
    def fit_model_with_ds(self, ds):
        self.fit(ds, callbacks=self.img_callback)
        
    def fit_model_with_array(self, x, y):
        self.fit(x, y, callbacks=self.img_callback)

print(tf.__version__)        

# Both do not work
non_ds_model = random_model("non_ds") 
non_ds_model.compile_model()
non_ds_model.fit_model_with_array(inp_arr_without_ds, inp_arr_without_ds)
            
tf_ds_model = random_model("tf_ds")
tf_ds_model.compile_model()
tf_ds_model.fit_model_with_ds(tf_ds)