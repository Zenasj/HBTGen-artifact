from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
print('Tensorflow version', tf.__version__)

def input_fn(training=False):
    batch_size = 8
    def preprocess_map_func(image, label):
        image = tf.image.resize(image, size=[299, 299])
        image.set_shape([None, None, 3])
        image /=  127.5
        image -= 1
        return image, label
    
    def input_():
        if training:
            dataset = tfds.load(name='cats_vs_dogs', as_supervised=True, split=["train"])[0]
            train_dataset = dataset.skip(3000)
            train_dataset = train_dataset.map(preprocess_map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_dataset = train_dataset.shuffle(1024).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
            return train_dataset
        else:
            dataset = tfds.load(name='cats_vs_dogs', as_supervised=True, split=["train"])[0]
            test_dataset = dataset.take(3000)
            test_dataset = test_dataset.map(preprocess_map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.shuffle(1024).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
            return test_dataset
    return input_

epochs = 10
samples = 23000
batch_size = 8

base_model = Xception(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
y = GlobalAveragePooling2D()(base_model.output)
y = Dense(units=1, activation='linear', kernel_initializer='he_normal')(y)
base_model.trainable = False
model = tf.keras.Model(base_model.input, y)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)
train_spec = tf.estimator.TrainSpec(input_fn(training=True), max_steps=epochs * samples//batch_size)
eval_spec = tf.estimator.EvalSpec(input_fn(training=False), steps=3000//batch_size)
tf.estimator.train_and_evaluate(estimator, 
                                train_spec=train_spec, 
                                eval_spec=eval_spec)