import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


def call(q, v, k, mask_q=None, mask_v=None):
    """ Call attention instance """
    return attn(inputs=[q, v, k], mask=[mask_q, mask_v])

x = tf.random.uniform((1, 2, 2))
attn = tf.keras.layers.Attention(use_scale=True)

# position arguments work well
call(x, x, x)

# naming the parameters also fine here
call(q=x, v=x, k=x)

class MyAttention(tf.keras.Model):
    
    def __init__(self):
        super(MyAttention, self).__init__()
        self.attention = tf.keras.layers.Attention(use_scale=True)
        
    def call(self, q, v, k, mask_q=None, mask_v=None):
        return self.attention(inputs=[q, v, k], mask=[mask_q, mask_v])


my_attention = MyAttention()

# Still works with positional arguments
my_attention(x, x, x)

# Breaks when naming the arguments in my_attention:
my_attention(q=x, v=x, k=x)

my_attention.call(q=x, v=x, k=x)

class MyAttention(tf.keras.Model):
    
    def __init__(self):
        super(MyAttention, self).__init__()
        self.attention = Attention(use_scale=True)
        
    def call(self, q, v, k, mask_q=None, mask_v=None, **kwargs):
        """ Print **kwargs, then call tf.keras.layers.Attention """
        for key, value in kwargs.items(): 
            print(f'{key} == {value}') 
        return self.attention(inputs=[q, v, k], mask=[mask_q, mask_v])

# as expected:
my_attention(x, x, x, extra_arg='hi')

# fails to print the kwarg, complains about missing positional arg `inputs`
my_attention(q=x, v=x, k=x, extra_arg='hi')

# it only takes leaving out the name of the first parameter for it to work again
my_attention(x, v=x, k=x, extra_arg='hi')