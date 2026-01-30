from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

###imports and default settings
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import sys,os
import keras.backend as K
slim = tf.contrib.slim
img_size=10
datas = np.array([np.ones((100)),np.zeros((100))])
labels = np.array([[1.0,0.0],[0.0,1.0]])
lr = 1e-3
beta_1=0.9
beta_2=0.999
epsilon=1e-08
seed=10
relu_function = tf.nn.relu

##keras model

tf.reset_default_graph()
with tf.get_default_graph().as_default() as keras_graph:
    with tf.Session(graph=keras_graph).as_default() as keras_sess:
        init = keras.initializers.RandomNormal(seed=seed)
        inputs = keras.layers.Input(shape=(100,))
        x = keras.layers.Dense(units=2,kernel_initializer=init,bias_initializer=init)(inputs)
        pred = keras.layers.Activation(activation='softmax')(x)
        model = keras.models.Model(inputs,pred)
        
        #tf optimizer
#         optimizer = tf.train.AdamOptimizer(lr,beta1=beta_1,beta2=beta_2,epsilon=epsilon)

        ###keras optimizer
        optimizer = keras.optimizers.Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)

        #use differnet loss 
        model.compile(optimizer=optimizer,loss=keras.losses.categorical_crossentropy)

        #get first cnn layer weights
        keras_variable_name = [x.name for x in tf.trainable_variables()[:2]]
        keras_each_layer_tensors = tf.trainable_variables()[:2]
        keras_weights_list=[]
        keras_gradients_list = []
        ####gradient
        #get gradient tensor
        keras_gradients_tensor = model.optimizer.get_gradients(loss=model.loss(model.targets[0],model.outputs[0]),
                                                        params=keras_each_layer_tensors)
        
        keras_sess = keras.backend.get_session()
        #get layer weights before training
        keras_weights_list.append(keras_sess.run(keras_each_layer_tensors))
        #get gradients before training
        keras_gradients_list.append(keras_sess.run(keras_gradients_tensor,feed_dict={model.inputs[0]:datas,model.targets[0]:labels}))
        #### update weights
        model.fit(x=datas,y=labels,epochs=1,verbose=2)
        #get layer weights after 1 epoch training
        keras_weights_list.append(keras_sess.run(keras_each_layer_tensors))
        #get gradients after 1 epoch training
        keras_gradients_list.append(keras_sess.run(keras_gradients_tensor,feed_dict={model.input:datas,model.targets[0]:labels}))

        keras_weights_list = np.array(keras_weights_list)
        keras_gradients_list = np.array(keras_gradients_list)
        keras_weights_update = keras_weights_list[1]-keras_weights_list[0]

###### tf model
tf.reset_default_graph()
with tf.get_default_graph().as_default() as tf_graph:
    with tf.Session(graph=tf_graph).as_default() as tf_sess:
        initializer = tf.initializers.random_normal(seed=seed)
        
        ##placeholders
        tf_input = tf.placeholder(tf.float32, [None, 100],
                    name='input')
        tf_label = tf.placeholder(tf.float32,[None,2],name='label')
        tf_lr = tf.placeholder(tf.float32,[],name='lr')
        #model
        x = tf.layers.dense(inputs=tf_input,units=2,kernel_initializer=initializer,bias_initializer=initializer)
        pred = tf.nn.softmax(x)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_label,logits=x)
#         loss = keras.losses.categorical_crossentropy(y_pred=pred,y_true=tf_label)
        tf_loss = tf.reduce_mean(loss)
#         tf_optimizer = tf.keras.optimizers.Adam(lr=lr)
        tf_optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta_1,beta2=beta_2,epsilon=epsilon,name='optimizer')
        train = tf_optimizer.minimize(loss=tf_loss)
        
        #get tensors
        tf_each_layer_tensors = tf.trainable_variables()
        tf_variable_name = [x.name for x in tf.trainable_variables()]
        #get gradient tensor
#         tf_gradient_tensor  = tf.gradients(tf_loss,tf_each_layer_tensors)
        tf_gradient_tensor  = keras.backend.gradients(tf_loss,tf_each_layer_tensors)
        #init
        tf_sess.run(tf.global_variables_initializer())
        
        tf_weights_list=[]
        tf_gradients_list=[]
        #get init weight before training
        tf_weights_list.append(tf_sess.run(tf_each_layer_tensors))
        
        tf_gradients_list.append(tf_sess.run(tf_gradient_tensor,feed_dict={tf_label:labels,tf_input:datas}))

        loss_value,_ = tf_sess.run([tf_loss,train],feed_dict={tf_input:datas,tf_label:labels})
        print('loss:',loss_value)
        
        tf_weights_list.append(tf_sess.run(tf_each_layer_tensors))
        
        tf_gradients_list.append(tf_sess.run(tf_gradient_tensor,feed_dict={tf_label:labels,tf_input:datas}))
        
        tf_weights_list = np.array(tf_weights_list)
        tf_gradients_list = np.array(tf_gradients_list)
        
        tf_weights_update = tf_weights_list[1]-tf_weights_list[0]