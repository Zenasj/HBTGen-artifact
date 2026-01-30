from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras.backend as keras_backend
import tensorflow.keras as keras

class MetaModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(5, input_shape=(1,))
        self.out = keras.layers.Dense(1)
    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = self.out(x)
        return x

def copy_model(model, x):
    copied_model = MetaModel()
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

def compute_loss(model, x, y):
    logits = model.forward(x)  # prediction of my model
    mse = keras_backend.mean(keras.losses.mean_squared_error(y, logits))  # compute loss between prediciton and label/truth
    return mse, logits

optimizer_outer = keras.optimizers.Adam()
alpha = 0.01
with tf.GradientTape() as g:
    # meta_model to learn in outer gradient tape
    meta_model = MetaModel()
    # inputs for training
    x = tf.constant(3.0, shape=(1, 1, 1))
    y = tf.constant(3.0, shape=(1, 1, 1))

    meta_model.forward(x)
    model_copy = copy_model(meta_model, x)
    with tf.GradientTape() as gg:
        loss, _ = compute_loss(model_copy, x, y)
        gradients = gg.gradient(loss, model_copy.trainable_variables)
        k = 0
        for layer in range(len(model_copy.layers)):
            """ If I use meta-model for updating, this works """
            # model_copy.layers[layer].kernel = tf.subtract(meta_model.layers[layer].kernel,
            #                                               tf.multiply(alpha, gradients[k]))
            # model_copy.layers[layer].bias = tf.subtract(meta_model.layers[layer].bias,
            #                                             tf.multiply(alpha, gradients[k + 1]))

            """ If I use model-copy for updating instead, gradients_meta always will be [None,None,...]"""
            model_copy.layers[layer].kernel = tf.subtract(model_copy.layers[layer].kernel,
                                                          tf.multiply(alpha, gradients[k]))
            model_copy.layers[layer].bias = tf.subtract(model_copy.layers[layer].bias,
                                                        tf.multiply(alpha, gradients[k + 1]))

            k += 2

    # calculate loss of model_copy
    test_loss, _ = compute_loss(model_copy, x, y)
    # build gradients for meta_model update
    gradients_meta = g.gradient(test_loss, meta_model.trainable_variables)
    """ gradients always None !?!!11 elf """
    optimizer_outer.apply_gradients(zip(gradients_meta, meta_model.trainable_variables))