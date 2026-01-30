import numpy as np
import tensorflow as tf

@tf.custom_gradient
def select_subnet_layer(x):
    # the last in x are the select, before is divided into parts
    x_select = x[:,:,-args.lstm_num:]
    x_data = x[:,:,:-args.lstm_num]
    print("x_select",x_select.shape)
    size_of_out = x_data.shape[2] // args.lstm_num
    out = x_data[:,:,:size_of_out]
    out = 0 * out
    for i in range(args.lstm_num):
        out += x_data[:,:,(i * size_of_out):((i+1)*size_of_out)]* x_select[:,:,i:i+1]
    print("out",out.shape)
    print("x",x.shape)
    def custom_grad(dy):
        size_of_out = (x.shape[2]-args.lstm_num) // args.lstm_num
        gg = []
        gs = []
        for i in range(args.lstm_num):
            gg.append(  x[:,:,size_of_out * args.lstm_num +i:size_of_out * args.lstm_num + i + 1] * dy)
            tmp =  x[:,:,size_of_out * i:size_of_out * (i+1)] * dy
            gs.append(keras.backend.sum(tmp, axis = 2, keepdims = True))
        grad = keras.backend.concatenate(gg + gs)
        print(gg,gs,grad)
        return grad # keras.backend.clip(grad,-1,1)
    return out, custom_grad

@tf.custom_gradient
def select_subnet_layer(x):
    global ssss
    # the last in x are the select, before is divided into parts
    x_select = x[:,:,-args.lstm_num:]
    x_data = x[:,:,:-args.lstm_num]
    print("x_select",x_select.shape)
    size_of_out = x_data.shape[2] // args.lstm_num
    out = x_data[:,:,:size_of_out]
    out = 0 * out
    for i in range(args.lstm_num):
        out += x_data[:,:,(i * size_of_out):((i+1)*size_of_out)]* x_select[:,:,i:i+1]
    print("out",out.shape)
    print("x",x.shape)
    def custom_grad(dy):
        print('debugging',dy)
        s1 = dy.shape.as_list()[0]
        s2 = dy.shape.as_list()[1]
        print(dy,[dy])
        if s1 is None:
            return tf.fill((1, 324, size_of_out*args.lstm_num + args.lstm_num), 1.0)
        grad_nump = np.ones([s1,s2,153], dtype='float32')
        if x.shape.as_list()[0] is not None:
            for i in range(args.lstm_num):
                print('???',args.lstm_num)
                grad_nump[:,:, size_of_out*i : size_of_out*(i+1)] = x[:,:,size_of_out * args.lstm_num + i]
                grad_nump[:,:,size_of_out * args + i] = np.sum(x[:,:,size_of_out*i : size_of_out*(i+1)])
        grad = tf.convert_to_tensor(grad_nump)
        return grad
    return out, custom_grad

class CustomLayer(Layer):

    def __init__(self, **kwargs):

        super(CustomLayer, self).__init__(**kwargs)

    def call(self, x):
        return select_subnet_layer(x[:,:])  # you don't need to explicitly define the custom gradient

    def compute_output_shape(self, input_shape):
        print(input_shape[2])
        return (input_shape[0], input_shape[1], (int(input_shape[2]) - args.lstm_num ) // args.lstm_num)