import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class Brownian(gpflow.kernels.Kernel):
    def __init__(self,**kwargs):
        super().__init__()
        self._dtype = np.float64
        dtype = kwargs.get('dtype', None)
        
        self._feature_ndims = [0]
        #feature_ndims = kwargs.get('feature_ndims', None)
        #self.variance = gpflow.Parameter(1.0, transform=positive(),dtype=dtype)
        self.H = gpflow.Parameter(0.85, transform=positive(),dtype=dtype)#tf.transpose(X2)
        
    def K(self, x, X2=None):
      if X2 is None:
        X2 = X
        v = (np.absolute(x))**(2*self.H)+(np.absolute(tf.transpose(X2)))**(2*self.H) - (np.absolute(x))-(np.absolute(tf.transpose(X2)))**(2*self.H)
        #1/2*(1+1**(2*self.H)-1-X**(2*self.H))
      return 1/2*v#.astype(x.dtype) # this returns a 2D tensor

    def K_diag(self, x):
      return(1/2 * tf.reshape(v, (-1,))).astype(x.dtype)            # this returns a 1D tensor
    
    @property
    def feature_ndims(self):
      return tf.keras.backend.ndim(x)


    @property
    def dtype(self):
      #DType over which the kernel operates.
      return self._dtype
      
    @property
    def kernel(self):
      return (1/2 * tf.reshape(x, (-1,)))#.astype(x.dtype)  

    
    """
    @property
    def kernels(self):
    The list of kernels this _SumKernel sums over.
      return self._kernels
    """
#k_brownian = Brownian(dtype=x.dtype)
import sys
sys.setrecursionlimit(10000)

x_tst = x[189::]
x_range = 237
num_distributions_over_Functions = 1
tf.keras.backend.set_floatx('float64')
#kernel = Brownian #tfp.positive_semidefinite_kernels.ExponentiatedQuadratic#MaternOneHalf()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,14), dtype=np.float64),
    tf.keras.layers.LSTM(25,kernel_initializer='ones',activation='tanh', dtype = x.dtype, use_bias=True),
    #tf.keras.layers.InputLayer(input_shape=(10),dtype=x.dtype),#put a 1 before the 9 later
    tf.keras.layers.Dense(50,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(75,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(125,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(175,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(250,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
    #goal is to eventually replace the first dense layer with an LSTM layer
    #tf.keras.layers.LSTM
    #tf.keras.layers.TimeDistributed(Dense(vocabulary)))
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(125,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(75,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(25, kernel_initializer='ones',use_bias=False,),
    tfp.layers.VariationalGaussianProcess(
    num_inducing_points=num_inducing_points, kernel_provider=RBFKernelFn(dtype=x.dtype) , event_shape=(1,),
    inducing_index_points_initializer=tf.compat.v1.constant_initializer(
            np.linspace(0,x_range, num=1125,
                        dtype=x.dtype)[..., np.newaxis]), unconstrained_observation_noise_variance_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(x.dtype))),variational_inducing_observations_scale_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(np.float64))), mean_fn=None,
    jitter=1e-06, convert_to_tensor_fn=tfp.distributions.Distribution.sample)

  
    #in unconstrained thing replace astype with tf.dtype thing.  RBFKernelFn(dtype=x.dtype)  Brownian(dtype=x.dtype)#tf.initializers.constant(-10.0)
])

loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
#tf.keras.optimizers.Adam(1e-4) tf.optimizers.Adam(learning_rate=0.011)
model.compile(optimizer=tf.keras.optimizers.Adam(0.011), loss=loss)#tf.optimizers.Adam(learning_rate=0.01)
model.fit(x, y,epochs=370, verbose=True,validation_split=0.2)