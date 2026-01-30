from tensorflow.keras import layers
from tensorflow.keras import models

#!/usr/bin/env python3
from keras.layers import Layer
import numpy as np
from keras.layers import Input
from keras.models import Model  

InputTensor = [0.1,0.2,0.3]      # Attention! Different tensors cause errors. 
                                 # See  below the code!

InputTensor = np.array(InputTensor) 

class MyLayer(Layer):
    
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.built = True
        super(MyLayer,self).build(input_shape)

    def call(self, x):
        return x

inputlayer = Input(shape = InputTensor.shape)
layer = MyLayer()(inputlayer)

model = Model(inputlayer, layer ) 
model.summary()

Output = model.predict(InputTensor)
output = np.array(Output)
print("dim(Output) = "+str(output.ndim))
print("Output = "+str(Output))

class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.built = True
        super(MyLayer,self).build(input_shape)

    def call(self, x):
        OutputTensor = [[0.0, 0.0],
                        [0.0, 0.0]]
        OutputTensor[0][0] = x[0][0]
        OutputTensor[1][1] = x[1][0]
        OutputTensor[1][0] = x[2][0]
        return OutputTensor