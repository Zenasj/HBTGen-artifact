import tensorflow as tf

def max_pool_with_argmax(x):
    """According to the documentation, the value at each position in the argmax tensor 
    is calculated according to:  so that a maximum value at position [b, y, x, c] 
    becomes flattened index: (y * width + x) * channels + c if include_batch_in_index 
    is False; ((b * height + y) * width + x) * channels + c if include_batch_in_index is True.
    
    """
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class UnpoolingLayer(Layer):
    
    """ This class generates the Unpooling Layer present in the Segnet and 
    DeconvNet models.
    """
    def __init__(self, pooling_argmax, stride=[1,2,2,1], **kwargs):
        self.stride = stride
        self.pooling_argmax = pooling_argmax
        super(UnpoolingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(UnpoolingLayer, self).build(input_shape)

        
    def call(self, inputs):
       input_shape = K.cast(K.shape(inputs), dtype='int64')  # Convert
       
       output_shape = (input_shape[0],
                       input_shape[1]*self.stride[1],
                       input_shape[2]*self.stride[2],
                       input_shape[3])
       
       argmax = self.pooling_argmax
       one_like_mask = K.ones_like(argmax) # Create Tensor of 1s with same shape as argmax --> 4-dimensional tensor
       batch_range = K.reshape(K.arange(start=0, stop=input_shape[0], dtype='int64'), 
                                 shape=[input_shape[0], 1, 1, 1]) # Create a tensor of shape (Batch Size, 1 ,1 ,1)
       
       b = one_like_mask * batch_range  # 4 dimensional tensor
       #Multiply the ones mask by the batch range, so that we have a 4-dimension tensor, wth the pixels in the first dimension indicating the batch id for each index
       y = argmax // (output_shape[2] * output_shape[3])
       x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
       feature_range = K.arange(start=0, stop=output_shape[3], dtype='int64')  # Indicate the channel index
       f = one_like_mask * feature_range
       
       updates_size = tf.size(inputs)  # Number of elements in the tensor
       indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))   # Generate a 2D array where the rows are the b, y, x , f values and the columns are actually the number of elements in the input tensor, and then just transpose it
       values = K.reshape(inputs, [updates_size]) # flatten it to one dimension so that they can be feed to the tf.scatter
       
       
       return tf.scatter_nd(indices, values, output_shape)
   
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])
    
    
    def get_config(self):
        base_config = super(UnpoolingLayer, self).get_config()
        base_config['pooling_argmax'] = self.pooling_argmax
        base_config['stride'] = self.stride
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
       

if __name__ == "__main__":
    
    
   input_tensor = Input(shape=(128,128,1))
   
   pool1, pool1_argmax = Lambda(max_pool_with_argmax, name='max_pool1')(input_tensor)
   
   x = Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', name='stage1_conv1')(pool1)
   
   unpool1 = UnpoolingLayer(pool1_argmax, name='unpool1')(x)
   unpool1.set_shape(pool1.get_shape())
   
   x = Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', name='stage1_conv1')(unpool1)
   
   model = Model(inputs = input_tensor, outputs = unpool1)
   model.summary()