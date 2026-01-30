from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

class Model(tf.keras.Model):
    
    @property
    def groupedVariables(self):
        if self._var is None:
            self._var = []
            for denses in self._denses:
                self._var.append([])
                for d in denses:
                    self._var[-1] = self._var[-1] + d.trainable_variables                    
        return self._var
    ## ------------------------------------------------------------------------
    def __init__(self, ** kwargs):
        super(Model, self).__init__(** kwargs)

        self._optimizers = []
        self._denses     = []
        self._var        = None
        for copy in range(2):
            self._optimizers.append(tf.keras.optimizers.Adam())
            self._denses    .append([ tf.keras.layers.Dense(s) for s in [2,1]])
    ## ------------------------------------------------------------------------
    def call(self, x):
        y = []    
        for denses in self._denses:
            yy = x
            for d in denses: yy = d(yy)
            y.append(yy) 
        return y
    ## ------------------------------------------------------------------------
    def update(self, x, t):
        loss = []
        with tf.GradientTape() as tape:
            yy   = self(x)
            for y in zip(yy):
                y = y[0]
                l = tf.reduce_mean((t - y) ** 2)
                loss.append(l)

        var  = self.groupedVariables
        grad = tape.gradient(loss, var) 
        for g,v,o in zip(grad, var, self._optimizers):
            o.apply_gradients(zip(g,v))
## ----------------------------------------------------------------------------
m = Model()        
x = tf.zeros(shape = [1, 2], dtype = tf.float32)

print("Simple run then save ... ", end = "")
m(x)
m.save_weights("TMP/model") ## <== This works
print("done")

print("Update then save ....... ", end = "")
m.update(x, 0)
m.save_weights("TMP/model") ## <== This craches
print("done")