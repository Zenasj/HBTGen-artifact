import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

In [176]: sys.version
Out[176]: '3.7.3 (default, Mar 27 2019, 16:54:48) \n[Clang 4.0.1 (tags/RELEASE_401/final)]'

In [177]: tf.__version__
Out[177]: '2.0.0-alpha0'

python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np

m = 1000
n = 1
X = np.random.randn(m, n).astype(np.float32)
y = (3 + 0 * np.random.randn(m)).astype(np.float32)

def create_model():
    a_input = keras.layers.Input(shape=(n,), dtype=np.float32)
    a = K.expand_dims(a_input, axis=2)
    q = keras.layers.Conv1D(1, 1)(a)
    q = - tf.math.square(q) # this breaks things, but only when using tf.function
    model = keras.models.Model(inputs=a_input, outputs=q)
    return model

model = create_model()
model.predict(X)

class Trainer():
    def __init__(self, epochs=10):
        self.epochs = epochs
        self.model = create_model()
        self.optimizer = tf.optimizers.Adam()
        self.step = 0
    def train(self, X, y, epochs=10):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        for epoch in range(epochs):
            l = self._train_one_step(X, y)
        return l
    @tf.function
    def _train_one_step(self, X, y):
        with tf.GradientTape() as tape:
            yp = self.model(X)
            loss = tf.reduce_mean(tf.math.square(y - yp))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        l = self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        d = dict(loss=loss)
        tf.print(yp[0], loss)
        self.step += 1

trainer = Trainer()
l = trainer.train(X, y, epochs=100)

class Train(object):
  def __init__(self,epochs, enable_function, batch_size, per_replica_batch_size):
    self.epochs = epochs
    self.enable_function = enable_function
    self.batch_size = batch_size
    self.per_replica_batch_size = per_replica_batch_size
    self.learning_rate =  CustomSchedule(10)
    self.model = MyModel()
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
     reduction=tf.keras.losses.Reduction.SUM)
    self.optimizer = tf.keras.optimizers.Adam()
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
  
  def loss_function(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = self.loss_object(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      return tf.reduce_sum(loss_) * 1./self.batch_size

  def train_step(self, inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(images)
      loss = self.loss_function(labels, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    self.train_loss(loss)
    self.train_accuracy(labels, predictions)
    
  def test_step(self, inputs):
    images, labels = inputs
    predictions = self.model(images)
    t_loss = self.loss_function(labels, predictions)

    self.test_loss(t_loss)
    self.test_accuracy(labels, predictions)
    
  def training_loop(self, train_dataset, test_dataset):
    if self.enable_function:
      self.train_step=tf.function(self.train_step)
      self.test_step=tf.function(self.test_step)
    for epoch in range(self.epochs):
      self.train_loss.reset_states()
      self.test_loss.reset_states()
      self.train_accuracy.reset_states()
      self.test_accuracy.reset_states()

      for images, labels in train_dataset:
        self.train_step((images, labels))
      for test_images,test_labels in test_dataset:
        self.test_step((test_images, test_labels))

      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print (template.format(epoch+1,
                             self.train_loss.result(),
                             self.train_accuracy.result()*100,
                             self.test_loss.result(),
                             self.test_accuracy.result()*100))
      
      
      
 
epochs=5
enable_function=True
batch_size=128
per_replica_batch_size=128
train_obj=Train(epochs, enable_function, batch_size, per_replica_batch_size)
train_obj.training_loop(train_ds, test_ds)
tf.saved_model.save(train_obj.model,'model')

class DistributedTrain(Train):
  def __init__(self,epochs, enable_function, batch_size, per_replica_batch_size):
    Train.__init__(self,epochs, enable_function, batch_size, per_replica_batch_size)
      
  def training_loop(self, train_iterator, test_iterator, 
                   num_train_steps_per_epoch, num_test_steps_per_epoch,
                   strategy):
    def distributed_train():
      return strategy.experimental_run(self.train_step, train_iterator)

    def distributed_test():
      return strategy.experimental_run(self.test_step, test_iterator)

    if self.enable_function:
      distributed_train = tf.function(distributed_train)
      distributed_test = tf.function(distributed_test)

    template = 'Epoch: {}, Train Loss: {}, Test Loss: {}'
    for epoch in range(self.epochs):
      self.train_loss.reset_states()
      self.test_loss.reset_states()
      self.train_accuracy.reset_states()
      self.test_accuracy.reset_states()

      train_iterator.initialize()
      for _ in range(num_train_steps_per_epoch):
        distributed_train()

      test_iterator.initialize()
      for _ in range(num_test_steps_per_epoch):
        distributed_test()

      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print (template.format(epoch+1,
                               self.train_loss.result(),
                               self.train_accuracy.result()*100,
                               self.test_loss.result(),
                               self.test_accuracy.result()*100))


strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync

with strategy.scope():
  num_train_steps_per_epoch = tf.data.experimental.cardinality(train_ds)
  num_test_steps_per_epoch = tf.data.experimental.cardinality(test_ds)

  train_iterator = strategy.make_dataset_iterator(train_ds)
  test_iterator = strategy.make_dataset_iterator(test_ds)
  
  train_obj= DistributedTrain(epochs, enable_function, batch_size, per_replica_batch_size)
  train_obj.training_loop(train_iterator,
                          test_iterator,
                                   num_train_steps_per_epoch,
                                   num_test_steps_per_epoch,
                                   strategy)
  tf.saved_model.save(train_obj.model,'dist-model')