from tensorflow import keras
from tensorflow.keras import optimizers

self.train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

import os
import numpy as np
import cv2
import tensorflow as tf



class ModelTrain():
    def __init__(self):
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.train_loss = tf.keras.metrics.CategoricalCrossentropy('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.CategoricalCrossentropy('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.CategoricalAccuracy('validation_accuracy')
        
if __name__ == "__main__":
    model_train = ModelTrain()