import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=10, output_dim=12, embeddings_initializer=keras.initializers.zeros))
model.compile(optimizer=keras.optimizers.SGD(),loss=keras.losses.MeanSquaredError())

x = numpy.append(numpy.zeros(10000), numpy.ones(10000))
y = numpy.append(numpy.random.multivariate_normal(numpy.zeros(12), numpy.diag(numpy.ones(12)), 10000),
                 numpy.random.multivariate_normal(numpy.ones(12)*2, numpy.diag(numpy.ones(12)), 10000), axis=0)
model.fit(x,y,epochs=1,batch_size=1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_shape=(10,), use_bias=False, kernel_initializer=keras.initializers.zeros))
model.compile(optimizer=keras.optimizers.SGD(),loss=keras.losses.MeanSquaredError())

x = tensorflow.one_hot(numpy.append(numpy.zeros(10000), numpy.ones(10000)), 10)
y = numpy.append(numpy.random.multivariate_normal(numpy.zeros(12), numpy.diag(numpy.ones(12)), 10000),
                 numpy.random.multivariate_normal(numpy.ones(12)*2, numpy.diag(numpy.ones(12)), 10000), axis=0)
model.fit(x,y,epochs=16,batch_size=128)