import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

## define custom loss
def negloglik_loss(y_true, y_pred):
    nll,_,_ = y_pred
    return nll
## define custom metric
def negloglik_metric(y_true, y_pred):
    nll,_,_ = y_pred
    return nll
    
    
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

devices_names = [d.name.split('e:')[1] for d in gpus]

print(devices_names)
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names)


## create dataset
w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  #x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  #x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_tst = load_dataset()

## build model from model class inheritance
class tfp_prob_reg(tf.keras.Model):
    
    def __init__(self):
        super(tfp_prob_reg, self).__init__()
        self.block_1 = tf.keras.layers.Dense(1, activation='relu')
        
    def call(self, inputs):
        input_x, input_y = inputs
        x_mu = self.block_1(input_x)
        dist = tfp.layers.DistributionLambda(lambda x_mu: tfd.Normal(loc=x_mu, scale=1))(x_mu)
        
        
        return -dist.log_prob(input_y), dist, x_mu

## run on gpus
with strategy.scope():
    ## define inputs
    input_x = tf.keras.Input(shape=(1,))
    input_y = tf.keras.Input(shape=(1,))

    ## define output
    outputs_x = tfp_prob_reg()([input_x, input_y])

    ## build model
    tfp_model = tf.keras.Model(inputs = [input_x, input_y], outputs=outputs_x)
    tfp_model.add_loss(negloglik_loss(input_y, outputs_x))
    tfp_model.add_metric(negloglik_metric(input_y, outputs_x), name='metric')

    ## compile model
    tfp_model.compile(optimizer=tf.optimizers.Nadam(learning_rate=1e-5))

## fit model
tfp_model.fit([x,y],
              epochs=10,
              validation_split = .1,
              verbose=True)