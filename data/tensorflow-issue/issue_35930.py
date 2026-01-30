import numpy as np
import tensorflow as tf

class Model(keras.Model):
    def __init__(self, inp1, inp2):
        super(Model, self).__init__()
        self.x1 = self.add_weight('w1',[inp1])
        self.x2 = self.add_weight('w2',[inp2])
    def call(self,x):
        return x
# load_weights method works when save format is 'tf
x = Model(100,200)
x.save_weights('temp.tmp',save_format='tf')
old = x.weights[0][0].numpy()
print(old)
x = Model(100,200)
x.load_weights('temp.tmp')
new = x.weights[0][0].numpy()
print(new)
print(old==new)

# load_weights method does not work when save format is 'h5'
x = Model(100,200)
x.save_weights('temp.h5',save_format='h5')
old = x.weights[0][0].numpy()
print(old)
x = Model(100,200)
x.load_weights('temp.h5')
new = x.weights[0][0].numpy()
print(new)
print(old==new)

class Model(keras.Model):
    def __init__(self, inp1, inp2):
        super(Model, self).__init__()
        self.x1 = self.add_weight('w1',[inp1])
        self.x2 = self.add_weight('w2',[inp2])
    def call(self,x):
        return x
# load_weights method works when save format is 'tf
x = Model(100,200)
x.save_weights('temp.tmp',save_format='tf')
old = x.weights[0][0].numpy()
x.weights[0].assign(tf.zeros_like(x.weights[0]))
print(old)
x.load_weights('temp.tmp')
new = x.weights[0][0].numpy()
print(new)
print(old==new)

# load_weights method does not work when save format is 'h5'
x = Model(100,200)
x.save_weights('temp.h5',save_format='h5')
old = x.weights[0][0].numpy()
x.weights[0].assign(tf.zeros_like(x.weights[0]))
print(old)
x.load_weights('temp.h5')
new = x.weights[0][0].numpy()
print(new)
print(old==new)

def load_weights(model,save_path):
    hf = h5py.File(save_path, 'r')
    for i in model.trainable_weights:
        res = hf.get(i.name)
        res = tf.convert_to_tensor(np.array(res))
        if res.shape == i.shape:
            i.assign(res)