import random

import numpy as np 
import tensorflow as tf

path_save = '/home/mathewsa/stored_models/' #custom path to save network
save_model = str(path_save)+"test_save.ckpt"
end_it = 1000 #number of iterations
frac_train = 1.0 #randomly sampled fraction of data to create training set
frac_sample_train = 0.01 #randomly sampled fraction of data from training set to train in batches
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

#Generate training data
len_data = 10000
x_x = np.array([np.linspace(0.,1.,len_data)])
x_y = np.array([np.linspace(0.,1.,len_data)]) 
y_true = np.array([np.linspace(-1.,1.,len_data)])

N_train = int(frac_train*len_data)
idx = np.random.choice(len_data, N_train, replace=False)

x_train = x_x.T[idx,:]
y_train = x_y.T[idx,:] 
v1_train = y_true.T[idx,:] 

sample_batch_size = int(frac_sample_train*N_train)

np.random.seed(1234)
tf.set_random_seed(1234)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class NeuralNet:
    def __init__(self, x, y, v1, layers):
        X = np.concatenate([x, y], 1)  
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2] 
        self.v1 = v1 
        self.layers = layers 
        self.weights_v1, self.biases_v1 = self.initialize_NN(layers) 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False)) 
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]]) 
        self.v1_tf = tf.placeholder(tf.float32, shape=[None, self.v1.shape[1]])  
        self.v1_pred = self.net(self.x_tf, self.y_tf) 
        self.loss = tf.reduce_mean(tf.square(self.v1_tf - self.v1_pred)) 
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.weights_v1+self.biases_v1,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam_v1 = self.optimizer_Adam.minimize(self.loss, var_list=self.weights_v1+self.biases_v1) 
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()  
        self.sess.run(init)
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b) 
        return weights, biases
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim)) 
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y
    def net(self, x, y): 
        v1_out = self.neural_net(tf.concat([x,y], 1), self.weights_v1, self.biases_v1)
        v1 = v1_out[:,0:1]
        return v1
    def callback(self, loss):
        global Nfeval
        print(str(Nfeval)+' - Loss in loop: %.3e' % (loss))
        Nfeval += 1
    def fetch_minibatch(self, x_in, y_in, v1_in, N_train_sample):  
        idx_batch = np.random.choice(len(x_in), N_train_sample, replace=False)
        x_batch = x_in[idx_batch,:]
        y_batch = y_in[idx_batch,:] 
        v1_batch = v1_in[idx_batch,:] 
        return x_batch, y_batch, v1_batch
    def train(self, end_it): 
        saver = tf.train.Saver()
        print('Stage 4.20')
        try:
            saver.restore(self.sess, save_model) 
            print('Using previous model')
        except:
            self.Nfeval = 1
            print('No previous model') 
        it = 0
        while it < end_it: 
            x_res_batch, y_res_batch, v1_res_batch = self.fetch_minibatch(self.x, self.y, self.v1, sample_batch_size) # Fetch residual mini-batch
            tf_dict = {self.x_tf: x_res_batch, self.y_tf: y_res_batch,
                       self.v1_tf: v1_res_batch}
            self.sess.run(self.train_op_Adam_v1, tf_dict)
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback) 
            it = it + 1
        self.save_path = saver.save(self.sess, save_model)
        print('Finishing up training and saving as: ') 
        print(save_model) 
    def restore_model(self, path_full_saved):
        saver = tf.train.Saver()
        print('Stage 4.20')
        try:
            saver.restore(self.sess, str(path_full_saved))
            print('Using previous model')
        except:
            print('No previous model')
    def predict(self, x_star, y_star): 
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        v1_star = self.sess.run(self.v1_pred, tf_dict)  
        return v1_star

model = NeuralNet(x_train, y_train, v1_train, layers)
 
Nfeval = 1
model.train(end_it)