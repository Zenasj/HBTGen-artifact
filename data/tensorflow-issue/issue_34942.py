import random

import numpy as np
import tensorflow as tf
class ASRDataGenerator(object):
    def __init__(self,num):
        self.num = num
    def __call__(self):
        for i in range(self.num):
            for j in range(106):
                yield 'a','b',np.random.randn(100,120)

class TFASRDataSet(object):
    def __init__(self,num,batch_size):

        self.num = num
        self.batch_size = batch_size
        self.asrDataGenerator = ASRDataGenerator(num)
        
    def setDataSetIterator(self):
        
        dataset = tf.data.Dataset.from_generator(self.asrDataGenerator, (tf.string,tf.string,tf.float32))
        dataset = dataset.shuffle(30000)
        dataset = dataset.map(lambda s1,s2,feat: [s1,s2,feat])
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.iterator = dataset.make_initializable_iterator()
        
test_tfASRDataSet = TFASRDataSet(248,192)
test_tfASRDataSet.setDataSetIterator()
test_iter = test_tfASRDataSet.iterator
test_next = test_iter.get_next()   

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
run_config.allow_soft_placement = True

with tf.Session(config=run_config) as sess:

    for i in range(100):

        sess.run(test_iter.initializer)
        
        while True:
            try:
                loss_list = sess.run([test_next])
                print(len(loss_list[0]))
            except tf.errors.OutOfRangeError:
                print("train epoch %d finish" % (i+1))
                break

...
def view_used_mem():
  used_mem = psutil.virtual_memory().used
  print("used memory: {} Mb".format(used_mem / 1024 / 1024))


def main(argv):
  del argv

  test_tfASRDataSet = TFASRDataSet(248, 192)
  test_tfASRDataSet.setDataSetIterator()
  test_iter = test_tfASRDataSet.iterator
  test_next = test_iter.get_next()

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth = True
  run_config.allow_soft_placement = True

  with tf.Session(config=run_config) as sess:

    for i in range(100):

      sess.run(test_iter.initializer)

      while True:
        try:
          loss_list = sess.run([test_next])
        except tf.errors.OutOfRangeError:
          print('train epoch %d finish' % (i + 1))
          view_used_mem()
          break
...

import tensorflow as tf
import psutil

dataset = tf.Dataset.range(int(1e7))
iterator = dataset.shuffle(int(1e7)).batch(int(1e6))

for _ in iterator:
  used_mem = psutil.virtual_memory().used
  print("used memory: {} Mb".format(used_mem / 1024 / 1024))