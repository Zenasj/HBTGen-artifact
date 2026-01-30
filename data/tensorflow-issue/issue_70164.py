from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Input,Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import layers
from keras import regularizers

best_model_2 = tf.keras.Sequential([
    layers.Dense(181,activation='relu',input_shape=(21,)),
    tfp.layers.DenseVariational(181,activation='relu',make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=1/train_size,),
    layers.Dense(1,activation='linear')
])