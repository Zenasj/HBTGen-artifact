import tensorflow as tf

class MyModel(tf.Module):
  
  def __init__(self):
    super(MyModel, self).__init__()
    self._var = tf.Variable(1.)
    
model = MyModel()
saver = tf.train.Saver({"model": model})