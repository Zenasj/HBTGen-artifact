from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


class Convolution_Layer(tf.keras.layers.Layer):
    
    def __init__(self,kernel_height,kernel_width,channel_in,channel_out,stride,padding):
        super(Convolution_Layer,self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.initializer = tf.initializers.GlorotUniform()
        
        #weights:
        self.W = tf.Variable(self.initializer(shape = (kernel_height,kernel_width,channel_in,channel_out)))
        self.b = tf.Variable(self.initializer(shape = (channel_out,)))
        
    
    def call(self,x):
        x = tf.nn.conv2d(x,self.W,strides = self.stride,padding = self.padding) + self.b
        return x
    
    
class Batch_Normalization(tf.keras.layers.Layer):
    def __init__(self,depth,decay,convolution):
        super(Batch_Normalization,self).__init__()
        self.mean = tf.Variable(tf.constant(0.0,shape = [depth]),trainable = False)
        self.var = tf.Variable(tf.constant(1.0,shape = [depth]),trainable = False)
        self.beta = tf.Variable(tf.constant(0.0,shape = [depth]))
        self.gamma = tf.Variable(tf.constant(1.0,shape = [depth]))
        #exponentiall moving average object
        self.mov_avg = tf.train.ExponentialMovingAverage(decay = decay)
        self.epsilon = 0.001
        self.convolution = convolution
        
    def call(self,x,training = True):
        
        if training:
            if self.convolution:
                batch_mean,batch_var = tf.nn.moments(x, axes=[0, 1, 2],keepdims = False)
            else:
                batch_mean,batch_var = tf.nn.moments(x, axes=[0],keepdims = False)

            as_mean = self.mean.assign(batch_mean)
            as_variance = self.var.assign(batch_var)
            #ensured argument to be evaluated before anything you define in the with block
            with tf.control_dependencies([as_mean,as_variance]):
                ma =self.mov_avg.apply([self.mean,self.var])
                x = tf.nn.batch_normalization(x = x,mean = batch_mean,variance = batch_var,offset = self.beta,scale = self.gamma,variance_epsilon = self.epsilon)
                
        else:
            mean = self.mov_avg.average(self.mean)
            var = self.mov_avg.average(self.var)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            x = tf.nn.batch_normalization(x,mean,var,local_beta,local_gamma,self.epsilon)
            
        return x
            


class MaxPool(tf.keras.layers.Layer):
    def __init__(self,kernel_size,strides,padding):
        super(MaxPool,self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        
    def call(self,x):
        return tf.nn.max_pool2d(x,ksize = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.strides,self.strides,1],padding = self.padding)



class Dense_layer(tf.keras.layers.Layer):
    def __init__(self,dim_out):
        super(Dense_layer,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        self.dim_out = dim_out
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.dim_out),initializer = self.initializer,trainable=True,name='w')
        self.b = self.add_weight(shape=(self.dim_out,), initializer = self.initializer, trainable=True,name='b')
        
    def call(self,x):
        return x @ self.W + self.b
    

class Global_Average_Pooling(tf.keras.layers.Layer):
    def __init__(self,axis):
        super(AvgPooling,self).__init__()
        self.axis = axis


    def call(self,x):
        return tf.reduce_mean(x, axis = self.axis)
    


class Flatten_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Flatten_layer,self).__init__()
        
    def call(self,x,shape = False):
        return tf.reshape(x, [x.shape[0],-1])
 


class Softmax_layer(tf.keras.layers.Layer):
    def __init__(self,dim_out):
        super(Softmax_layer,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        self.dim_out = dim_out
        
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.dim_out),initializer = self.initializer,trainable=True,name='w_s')
        self.b = self.add_weight(shape=(self.dim_out,), initializer = self.initializer, trainable=True,name='b_s')
        
    def call(self,x):
        return tf.nn.softmax(tf.matmul(x,self.W) + self.b)



class AvgPooling(tf.keras.layers.Layer):
    def __init__(self,kernel_height,kernel_width,strides,padding):
        super(AvgPooling,self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.strides = strides
        self.padding = padding
        
        
    def call(self,x):
        return tf.nn.avg_pool(input = x,ksize = [self.kernel_height,self.kernel_width],strides = self.strides,padding = self.padding)



class Dropout_layer(Layer):
    def __init__(self,rate):
        super(Dropout_layer,self).__init__()
        self.rate = rate

    def call(self,x,training = None):
        if training:
            return tf.nn.dropout(x,self.rate)
        return x
    
    
class Identity(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(Identity,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        
        #Block one
        self.conv_1 = Convolution_Layer(1,1,filters[0],filters[1],1,padding = 'VALID')
        self.batch_norm_1 = Batch_Normalization(filters[1],0.99,convolution = True)

        
        #Block two
        self.conv_2 = Convolution_Layer(3,3,filters[1],filters[2],1,padding = 'SAME')
        self.batch_norm_2 = Batch_Normalization(filters[2],0.99,convolution = True)

        #Block two
        self.conv_3 = Convolution_Layer(1,1,filters[2],filters[3],1,padding = 'VALID')
        self.batch_norm_3 = Batch_Normalization(filters[3],0.99,convolution = True)
        
        #Dimension adjustment variable:
        #self.dimension = Convolution_Layer(1,1,channel_in,channel_out,1,padding = 'valid')
        
    def call(self,x,training = None):
        #Block one
        fx = self.conv_1(x)
        fx = self.batch_norm_1(fx,training)
        fx = tf.nn.relu(fx)
        
        #Block two
        fx = self.conv_2(fx)
        fx = self.batch_norm_2(fx,training)
        fx = tf.nn.relu(fx)

        #Block three
        fx = self.conv_3(fx)
        fx = self.batch_norm_3(fx,training)

        #add input:
        #fx = tf.nn.relu(fx + self.dimension(x))
        fx = tf.nn.relu(fx + x)
        
        return fx



class Convolution_Block(tf.keras.layers.Layer):
    def __init__(self,filters,stride):
        super(Convolution_Block,self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        
        #Block one
        self.conv_1 = Convolution_Layer(1,1,filters[0],filters[1],stride,padding = 'VALID')
        self.batch_norm_1 = Batch_Normalization(filters[1],0.99,convolution = True)
        
        #Block two
        self.conv_2 = Convolution_Layer(3,3,filters[1],filters[2],1,padding = 'SAME')
        self.batch_norm_2 = Batch_Normalization(filters[2],0.99,convolution = True)

        #Block three
        self.conv_3 = Convolution_Layer(1,1,filters[2],filters[3],1,padding = 'VALID')
        self.batch_norm_3 = Batch_Normalization(filters[3],0.99,convolution = True)
        
        #Dimension adjustment variable:
        self.dimension = Convolution_Layer(1,1,filters[0],filters[3],stride,padding = 'VALID')
        
    def call(self,x,training = None):
        #Block one
        fx = self.conv_1(x)
        fx = self.batch_norm_1(fx,training)
        fx = tf.nn.relu(fx)
        
        #Block two
        fx = self.conv_2(fx)
        fx = self.batch_norm_2(fx,training)
        fx = tf.nn.relu(fx)

        #Block three
        fx = self.conv_3(fx)
        fx = self.batch_norm_3(fx,training)

        #Skip connection
        fx = tf.nn.relu(fx + self.dimension(x))
        return fx
    
class Global_Average_Pooling(tf.keras.layers.Layer):
    def __init__(self,axis):
        super(Global_Average_Pooling,self).__init__()
        self.axis = axis


    def call(self,x):
        return tf.reduce_mean(x, axis = self.axis)
    


    
class Resnet(tf.keras.Model):

    def __init__(self,num_classes = 10):
        super(Resnet,self).__init__()

        #1st stage: convolution with max pooling

        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv_1 = Convolution_Layer(kernel_height = 7,kernel_width = 7,channel_in = 3,channel_out = 64,stride = 2,padding = 'VALID')
        self.batch_norm_1 = Batch_Normalization(depth = 64,decay = 0.99,convolution = True)
        self.max_pool_1 = MaxPool(kernel_size = 3,strides = 2,padding = 'VALID')

        '''
        self.identity_layers = []
        for f in identity_blocks_filters:
            #self.identity_layers.append(Identity(kernel_height = 3,kernel_width = 3,channel_in = f[0],channel_out = f[1],stride = 1,decay = 0.99))
        '''

        #2nd stage:
        self.block_2_res_conv_1 = Convolution_Block(filters = [64,64,64,256],stride = 1)
        self.block_2_res_ident_1 = Identity(filters = [256,64,64,256])
        self.block_2_res_ident_2 = Identity(filters = [256,64,64,256])


        #3rd stage:
        self.block_3_res_conv_1 = Convolution_Block(filters = [256,128,128,512],stride = 2)
        self.block_3_res_ident_1 = Identity(filters = [512,128,128,512])
        self.block_3_res_ident_2 = Identity(filters = [512,128,128,512])
        self.block_3_res_ident_3 = Identity(filters = [512,128,128,512])

        #4th stage:
        self.block_4_res_conv_1 = Convolution_Block(filters = [512,256,256,1024],stride = 2)
        self.block_4_res_ident_1 = Identity(filters = [1024,256,256,1024])
        self.block_4_res_ident_2 = Identity([1024,256,256,1024])
        self.block_4_res_ident_3 = Identity([1024,256,256,1024])
        self.block_4_res_ident_4 = Identity([1024,256,256,1024])
        self.block_4_res_ident_5 = Identity([1024,256,256,1024])

        #5th stage:
        self.block_5_res_conv_1 = Convolution_Block(filters = [1024,512,512,2048],stride = 2)
        self.block_5_res_ident_1 = Identity(filters = [1024,512,512,2048])
        self.block_5_res_ident_2 = Identity(filters = [1024,512,512,2048])



        self.global_avg_pool = Global_Average_Pooling(axis = [1,2])
        self.avg_pool = AvgPooling(kernel_height = 2,kernel_width = 2,strides = 1,padding = 'SAME')
        self.flatten = Flatten_layer()
        self.dense_1 = Dense_layer(512)
        self.softmax = Softmax_layer(num_classes)

    def call(self,x,training = None):

        #1st stage: convolution with max pooling
        x = self.zero_padding(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x,training)
        x = tf.nn.relu(x)
        x = self.max_pool_1(x)
        

        #2nd stage: 
        '''
        for i in range(len(self.identity_layers)):
            x = self.identity_layers[i](x)
        '''
        x = self.block_2_res_conv_1(x)
        x = self.block_2_res_ident_1(x)
        x = self.block_2_res_ident_2(x)


        #3rd stage:
        x = self.block_3_res_conv_1(x)
        x = self.block_3_res_ident_1(x)
        x = self.block_3_res_ident_2(x)
        x = self.block_3_res_ident_3(x)

        #4th stage:
        x = self.block_4_res_conv_1(x)
        x = self.block_4_res_ident_1(x) 
        x = self.block_4_res_ident_2(x) 
        x = self.block_4_res_ident_3(x) 
        x = self.block_4_res_ident_4(x) 
        x = self.block_4_res_ident_5(x) 

        #5th stage:
        x = self.block_5_res_conv_1(x) 
        x = self.block_5_res_ident_1(x) 
        x = self.block_5_res_ident_2(x) 

        
        x = self.global_avg_pool(x)
        #x = self.avg_pool(x)
        #x = self.flatten(x)
        x = self.dense_1(x)
        x = tf.nn.relu(x)
        
        x = self.softmax(x)
        
        return x

# Load Cifar-10 data-set
dataset_train, dataset_eval = tf.keras.datasets.cifar10.load_data()
dataset_train = tf.data.Dataset.from_tensor_slices(dataset_train)
dataset_eval = tf.data.Dataset.from_tensor_slices(dataset_eval)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


dataset_train = dataset_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(60000)
dataset_train = dataset_train.batch(256)
dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

dataset_eval = dataset_eval.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset_eval = dataset_eval.batch(256)
dataset_eval = dataset_eval.cache()
dataset_eval = dataset_eval.prefetch(tf.data.AUTOTUNE)



epochs = 5

#model = Resnet(len(class_names_eval))
model = Resnet(num_classes=10)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()#used in backprop


num_train_steps = len(dataset_train) * epochs

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

train_loss = tf.keras.metrics.Mean(name='train_loss')#mean of the losses per observation
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def training(X,y):
    with tf.GradientTape() as tape:#Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        predictions = model(X,training = True)
        loss = loss_object(y,predictions)

    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss(loss) 
    train_accuracy(y,predictions)



@tf.function
def testing(X,y):
    predictions = model(X,training = False)
    loss = loss_object(y,predictions)
    test_loss(loss)
    test_accuracy(y,predictions)





for epoch in range(epochs):
    for X,y in dataset_train:
        training(X,y)

    for X,y in dataset_eval:
        testing(X,y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))



  # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    
    test_accuracy.reset_states()


tf.saved_model.save(model, '/content/model')