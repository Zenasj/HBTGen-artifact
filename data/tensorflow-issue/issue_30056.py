from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import math
import numpy as np
import os
import shutil
import tensorflow as tf

### parameters
batch_size = 64
epochs = 10
weight_decay = 0.0005

def init_gpus(soft_device_placement=True, log_device_placement=False, create_virtual_devices=False, memory_limit=4096):

    tf.config.set_soft_device_placement(soft_device_placement)    
    tf.debugging.set_log_device_placement(log_device_placement)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # If there is only one GPU, create two logical virtual devices for developing
        # on a machine with only one GPU installed
        try:
            # Create 2 virtual GPUs on each physical GPU with the given memory_limit GPU memory
            if create_virtual_devices and len(gpus) == 1:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )

            else:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        # print out physical and logical GPUs
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    else:
        print("No visible GPU is detected...")

def prelu(x, name='default'):
    if name == 'default':
        return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    else:
        return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)

def center_loss(y_true, y_pred):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       Lc = 1/2 sum(|| xi - ci||)
    """
    return 0.5 * K.sum(y_pred, axis=0)

### model
def my_model(x, labels):
    # x = BatchNormalization()(x)
    #
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = prelu(x)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Dropout(0.25)(x)
    #
    x = Flatten()(x)
    x = Dense(2, kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x, name='side_out')
    #
    main = Dense(10, activation='softmax', name='main_out', kernel_regularizer=l2(weight_decay))(x)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, labels])
    return main, side

# Function for decaying the learning rate.
# You can define any decay function you need.
def lr_schedule(epoch):
    if epoch <= 5:
        learning_rate = 1e-3

    elif epoch <= 10:
        learning_rate = 1e-4

    else:
        learning_rate = 1e-5

    tf.summary.scalar('learning_rate', data=learning_rate, step=epoch)
    return learning_rate

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(
            name='centers',
            shape=(10, 2),
            initializer='uniform',
            trainable=False
        )

        super().build(input_shape)

    def call(self, x):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        Returns:
            A tensor or list/tuple of tensors.
        """
        features = x[0]
        labels = K.reshape(x[1], [-1])

        # get the tensor as specified in the label
        # the centers might repeate depending on the label index
        centers_batch = K.gather(self.centers, labels)

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = K.gather(unique_count, unique_idx)
        appear_times = K.reshape(appear_times, [-1, 1])
        #
        # center_loss_alfa default 0.5
        delta_centers = centers_batch - features
        delta_centers = delta_centers / tf.cast((1 + appear_times), tf.float32)
        delta_centers = self.alpha * delta_centers

        # scatter_sub does not support multi-gpu training, there is no equivalent operation 
        new_centers = tf.compat.v1.scatter_sub(self.centers, x[1], delta_centers)

        self.add_update((self.centers, new_centers), x)
        self.result = K.sum(K.square(features - centers_batch), axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def empty_dir(folder):
    """
    Empty a folder recursively.
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            print("Remove file: {}".format(file_path))
            os.remove(file_path)
        else:
            empty_dir(file_path)
            print("Remove folder: {}".format(file_path))
            os.rmdir(file_path)

def build_empty_dir(folder, root_dir=os.getcwd()):
    base_dir = os.path.join(root_dir, folder)
    os.makedirs(base_dir, exist_ok=True)
    empty_dir(os.path.join(root_dir, folder))

    return base_dir

"""
unset CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES="0"
python3 center_loss_mnist.py
"""

### run model
def run(lambda_centerloss):

    init_gpus(
        log_device_placement=False,
        create_virtual_devices=True
    )
    
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    ### get data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize to 0..1
    x_train, x_test = x_train/255, x_test/255
    x_train = np.float32(x_train);
    x_test  = np.float32(x_test)

    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    # reshape to matrix
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    ### compile
    main_input = Input((28, 28, 1))
    aux_input = Input((), dtype='int32')

    # Training using Multi-GPUs
    # 
    # tf.compat.v1.scatter_sub does not support multi-gpus training
    # with strategy.scope():
    #
    # The following exception with be thrown.
    #
    # AttributeError: 'Tensor' object has no attribute '_lazy_read'

    # comment out the following line for the training to run successfully
    with strategy.scope():
        final_output, side_output = my_model(main_input, aux_input)
        model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
        model.compile(
            optimizer='adam',
            loss=[losses.sparse_categorical_crossentropy, center_loss],
            metrics=['accuracy'],
            loss_weights=[1, lambda_centerloss]
        )
        model.summary()

    ### create the log directory
    log_dir = '/tmp/logs/' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    build_empty_dir(log_dir)

    # Initialize the file_writer for logging summary
    # create the file_writer to save events for Tensorboard
    summary_log_dir = log_dir + '/train'
    file_writer = tf.summary.create_file_writer(summary_log_dir)
    file_writer.set_as_default()    

    tb_callback = TensorBoard(log_dir=log_dir)

    # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/callbacks/LearningRateScheduler
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    ### fit
    dummy1 = np.zeros((x_train.shape[0], 1), dtype=int)
    dummy2 = np.zeros((x_test.shape [0], 1), dtype=int)

    #
    print('model.input[0].shape = ', model.input[0].shape)
    print('model.get_layer(\'side_out\').output.shape = ', model.get_layer('side_out').output.shape)
    #
    model.fit(
        [x_train, y_train],     # inputs =[main_input, aux_input]
        [y_train, dummy1 ],     # outputs=[final_output, side_output]
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=([x_test, y_test], [y_test, dummy2]),
        callbacks=[tb_callback, lr_callback]
    )
    # validation
    reduced_model = Model(inputs=model.input[0], outputs=model.get_layer('main_out').output)
    reduced_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    # evaluate
    eval_loss, eval_acc = reduced_model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=batch_size
    )
    print('\nEval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
    ### run training and val sets
    reduced_model = Model(inputs=model.input[0], outputs=model.get_layer('side_out').output)

    feats = reduced_model.predict(x_train)

    ### done
    K.clear_session()
    return

###
if __name__ == '__main__':
    run(0.0001)