from tensorflow.keras import layers

'''
Environment setup
>> conda create -n test_tf -c conda-forge python=3.9.12 tensorflow=2.8.0 cudnn=8.2.1.32
>> export CUDA_VISIBLE_DEVICES=""
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
from tensorflow import keras

def main():
    print('GPUs:', tf.config.list_physical_devices('GPU'))
    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    print('Output shape:', model.output_shape)

if __name__ == '__main__':
    main()