from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = tf.keras.layers.Dense(5)
        
    def __call__(self, x):
        return self.layer(x)
    
model = MyModel()

# Build the model
model(tf.zeros([3, 10]))

# this is something like a train_op that "changes" the content of the variable.
assign_op = model.variables[1].assign( tf.ones_like(model.variables[1]) )

# Let's create a session
sess = tf.train.SingularMonitoredSession()
#sess.run(tf.global_variables_initializer())

# we can see the 'bias' variable is initialized to ZERO
assert sess.run(model.variables[1]).mean() == 0.0

# Now let's make the 'bias' variable to all one...
sess.run(assign_op)

# sure it is ONE ...
assert sess.run(model.variables[1]).mean() == 1.0

# Let's save the Keras model parameter. The bias is set to ONE, right ?????
# Since model.save_weights try to create a new op (another bug #26430)
# and the graph has been finalized, we will 'unfinalize' the graph with a bit of hack
sess.graph._finalized = False
model.save_weights('/tmp/keras-one.h5')
sess.graph.finalize()


# Let's see what is stored in the model file ....
import h5py
h = h5py.File("/tmp/keras-one.h5")
assert h['dense']['dense']['bias:0'].value.mean() == 1.0     # <------ This will fail
# Actual output is: array([0., 0., 0., 0., 0.], dtype=float32)