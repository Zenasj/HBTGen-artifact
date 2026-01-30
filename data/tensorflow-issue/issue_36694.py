import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

def custom_loss2(output2, label2):
    loss_value = K.mean(binary_crossentropy(label2, output2)) 
    return loss_value
def EWC_loss(new_weights, old_weights, fisher_matrix, rate):
    sum_w = 0
    for v in range(len(fisher_matrix)):
        sum_w += tf.reduce_sum(tf.multiply(fisher_matrix[v], tf.square(new_weights[v] - old_weights[v]))) 
    return sum_w*rate

del optimizer
del tape
optimizer = tf.keras.optimizers.SGD()
ewc_model = tf.keras.models.clone_model(model)
old_weights = model.trainable_variables.copy()
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        out = ewc_model(features_t)
        new_weights = ewc_model.trainable_variables.copy()
        ewc_loss = EWC_loss(new_weights, old_weights, fisher_matrix, 0.5)
        loss = ewc_loss + custom_loss2(out, labels_2_t)
        grad = tape.gradient(loss, ewc_model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, ewc_model.trainable_variables))
    if (epoch+1)%100 == 0:
        print("epch: {}, loss: {}".format(epoch, loss.numpy()))
        print(ewc_loss.numpy(), loss.numpy())

del optimizer
del tape
optimizer = tf.keras.optimizers.SGD()
ewc_model = tf.keras.models.clone_model(model)
old_weights = model.trainable_variables.copy()
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        out = ewc_model(features_t)
        new_weights = ewc_model.trainable_variables.copy()
        ewc_loss = 0.5*EWC_loss(new_weights, old_weights, fisher_matrix, 1.0)
        loss = ewc_loss + custom_loss2(out, labels_2_t)
        grad = tape.gradient(loss, ewc_model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, ewc_model.trainable_variables))
    if (epoch+1)%100 == 0:
        print("epch: {}, loss: {}".format(epoch, loss.numpy()))
        print(ewc_loss.numpy(), loss.numpy())

tf.Tensor(14.144677747478463, shape=(), dtype=float64)
tf.Tensor(14.254150624518838, shape=(), dtype=float64)

tf.Tensor(0.15645654679566814, shape=(), dtype=float64)
tf.Tensor(0.263113701303186, shape=(), dtype=float64)

# del optimizer
# del tape

# Input layer, one hidden layer
input_layer = Input(batch_shape=(None, 20))
dense_1 = Dense(1028)(input_layer)
output_2 = Dense(1, activation="sigmoid")(dense_1)
model = Model(inputs=input_layer, outputs= output_2)
print(model.summary())

n_sample = 1000
fix = np.array([range(n_sample),]*20).transpose()
features = np.cos(fix + np.random.rand(n_sample,20))
labels_2 = np.cos(np.array([range(n_sample),]*1).transpose() + np.random.rand(n_sample,1))
labels_2 = np.array([labels_2>=0]).astype(float)