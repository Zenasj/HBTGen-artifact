def map_fun(filename):
	filename = filename.numpy()
	data = sio.loadmat(filename)
	x = tf.cast(data['X'],dtype = tf.float32)
	label = tf.cast(data['label'].reshape(-1,), dtype = tf.int8)
	return x,label

db = dataset.map(map_fun)

import tensorflow as tf
import scipy.io as sio
import glob
def map_fun(filename):
	with tf.compat.v1.Session().as_default() as sess:
		filename = tf.cast(filename,tf.string).eval(session = sess) #eval method need a default sess
	data = sio.loadmat(filename) #data.keys() = ['X','label'] 
	x = tf.cast(data['X'],dtype = tf.float32) # data['X'].shape = (num_timestamp,num_feature)
	label = tf.cast(data['label'].reshape(-1, ),dtype = tf.int8) # data['label'].shape = (num_timestamp,)
	return x, label
filelist = glob.glob('./<file_pattern>*.mat')
db = tf.data.Dataset.from_tensor_slices(filelist)
dataset = db.map(map_fun)

filename = next(iter(db))
# filename = <tf.Tensor: id=10, shape=(), dtype=string, numpy=b'test.mat'>
data1 = tf.io.read_file(filename)
# data1 = <tf.Tensor: id=11, shape=(), dtype=string, numpy=b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Dec  6 16:54:07 2019      \x00\x00\x00\x00\x00\x00\x00\x00\x00\x01IM\x0f\x00\x00\x00\xe8......
# type(data_tfio) = <class 'tensorflow.python.framework.ops.EagerTensor'>
data2 = scipy.io.loadmat(filename.numpy())
# data2 = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Dec  6 16:54:07 2019', '__version__': '1.0', '__globals__': [], 'X': array([[  7.8698893 , -10.556351  , ...],[...]]),'label': array([[2],[2],[2],...])
# data_scipy.keys() = dict_keys(['__header__', '__version__', '__globals__', 'X', 'label'])