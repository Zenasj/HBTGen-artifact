from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

model.save()

tf.keras.models.load_model()

model.save()

import os
import glob
import numpy as np
import tensorflow as tf
tf.__version__

gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus)

RESULT_DIR = os.path.join(os.getcwd(), 'Test', 'Results')
CHECKPOINT_FREQUENCY = 16
LOG_EVERY = 1

BATCH_SIZE_PER_GPU = 16
NUM_GPUS = len(gpus)
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS

def get_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, strides=1, kernel_size=(4,4), input_shape=(28,28,1)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    
    return model

class SparseCategoricalLoss(tf.keras.losses.Loss):
    
    def __init__(self, num_classes, name='SparseCategoricalLoss', from_logits=False, loss_weight=1.0, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.name = name
        self.from_logits=from_logits
        self.loss_weight = loss_weight
        
    def loss_fn(self, y_true, y_pred):
        label = y_true[:,0:self.num_classes]
        logit = y_pred[:,0:self.num_classes]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits,
                                                             name=self.name,
                                                             reduction=tf.keras.losses.Reduction.NONE)(label, logit)
        loss *= self.loss_weight
        return loss
    
    
    def call(self, y_true, y_pred):
        total_loss = self.loss_fn(y_true, y_pred)
        return total_loss

    def get_config(self):
         
        config = super().get_config().copy()
        config.update({
            'num_classes' : self.num_classes,
            'name' : self.name,
            'loss_weight' : self.loss_weight
        })
        return config

loss = SparseCategoricalLoss(num_classes=10,
                             from_logits=True,
                             name='categorical_loss')

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    
    model = get_model()
    
    optimizer = tf.keras.optimizers.RMSprop(
                                            learning_rate=0.001,
                                            epsilon=1.0,
                                            momentum=0.9,
                                            rho=0.9
                                           )
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)

class LoggingCallback(tf.keras.callbacks.Callback):

    def __init__(self, result_dir, log_every, initial_step=0, checkpoint_frequency=None, **kwargs):
        
        super().__init__(**kwargs)
        
        # Create result directory
        self.result_dir = result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # create checkpoint directory
        checkpoint_dir = os.path.join(self.result_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # create tensorboard directory
        tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')
        if not os.path.join(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        
        self.log_every = log_every
        self.checkpoint_frequency = checkpoint_frequency
        self.train_writer = tf.summary.create_file_writer( os.path.join(tensorboard_dir, 'train') )
        self.step = initial_step
        
        
    # Write metrics to TensorBoard    
    def write_metrics_tensorboard(self, logs):
        with self.train_writer.as_default():
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                tf.summary.scalar(name, value, step=self.step)
                
                
    def on_batch_end(self, batch, logs=None):
        
        self.step += 1
        
        # Write metrics to tensorboard
        if self.step % self.log_every == 0:
            self.write_metrics_tensorboard(logs)
            
        # Save model checkpoint (weights + optimizer state)
        if self.checkpoint_frequency and self.step % self.checkpoint_frequency == 0:
            name = 'model_step_%d.h5' % self.step
            path = os.path.join(self.result_dir, 'checkpoint', name)
            self.model.save( path )

callbacks = LoggingCallback(result_dir=RESULT_DIR, log_every=LOG_EVERY, checkpoint_frequency=CHECKPOINT_FREQUENCY)

model.fit(
          x = X_train, 
          y = Y_train, 
          batch_size=GLOBAL_BATCH_SIZE,
          epochs=7,
          validation_data = (X_test, Y_test),
          callbacks=callbacks,
          verbose=1 
         )

del model
del strategy

previous_checkpoints = glob.glob(os.path.join(RESULT_DIR, 'checkpoint', '*'))
previous_checkpoints.sort(key=lambda x : int(os.path.basename(x).split('_')[2].replace('.h5', '')) )
latest_checkpoint = previous_checkpoints[-1]
print('Found Latest Checkpoint : %s' % latest_checkpoint)
    
initial_step = int(os.path.basename(latest_checkpoint).split('_')[2].replace('.h5', ''))
print('Resuming training from step %d' % initial_step)
    
new_callback = LoggingCallback(result_dir=RESULT_DIR, log_every=LOG_EVERY, initial_step=initial_step, checkpoint_frequency=CHECKPOINT_FREQUENCY)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model( latest_checkpoint, custom_objects={'SparseCategoricalLoss':SparseCategoricalLoss} )

model.fit(
          x = X_train, 
          y = Y_train, 
          batch_size=GLOBAL_BATCH_SIZE,
          epochs=10,
          validation_data = (X_test, Y_test),
          callbacks=new_callback,
          verbose=1 
         )

model.fit()

model.save()

tf.distribute.Strategy