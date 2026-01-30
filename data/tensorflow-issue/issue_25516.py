import tensorflow as tf
from tensorflow.keras import optimizers

class Loss():
    def __init__(self):
        self.metrics = []
    
    def loss(self, y_true, y_pred):
        
        # the following can be some complicated intermediate stuff that
        # would otherwise require redundant code
        t1 = tf.reduce_mean(y_true-y_pred)
        m2 = tf.reduce_max(y_true-y_pred)
        m3 = tf.reduce_min(y_true-y_pred)
        
        def m1(y_true, y_pred):
            return t1
        self.metrics.append(m1)
        
        def make_fcn(tensor, name):
            f = lambda y_true, y_pred: tensor
            f.__name__ = name
            return f
        for m in ['m2', 'm3']:
            self.metrics.append(make_fcn(eval(m),m))
        
        return tf.reduce_sum((y_true-y_pred)**2)

output2_loss = Loss()

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001, decay=1e-6), 
    loss={'pred': 'categorical_crossentropy', 'output2': output2_loss.loss},
    metrics={'pred': ['accuracy'], 'output2': output2_loss.metrics}
)

class Loss():
    def __init__(self):
        self.metric_names = ['m1', 'm2', 'm3']
        
        def make_fcn(name):
            f = lambda y_true, y_pred: self._metric_functions[name](y_true, y_pred)
            f.__name__ = name
            return f
        self._metric_functions = {}
        self.metrics = [make_fcn(name) for name in self.metric_names]
    
    def loss(self, y_true, y_pred):
        
        # the following can be some complicated intermediate stuff that otherwise requires redundant code
        m1 = tf.reduce_mean(y_true-y_pred)
        m2 = tf.reduce_max(y_true-y_pred)
        m3 = tf.reduce_min(y_true-y_pred)
        
        def make_fcn(tensor):
            return lambda y_true, y_pred: tensor
        for name in self.metric_names:
            self._metric_functions[name] = make_fcn(eval(name))
        
        return tf.reduce_sum((y_true-y_pred)**2)

output2_loss = Loss()

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.001, decay=1e-6), 
    loss={'pred': 'categorical_crossentropy', 'output2': output2_loss.loss},
    metrics={'pred': ['accuracy'], 'output2': output2_loss.metrics}
)