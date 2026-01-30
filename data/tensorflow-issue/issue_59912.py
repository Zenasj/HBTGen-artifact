import tensorflow as tf
import numpy as np


a=np.ones([64])
b=np.ones([64])


class assign:
    def __init__(self):
        self.c=np.ones([64])
    
    
    def assign(self,d):
        for i in range(len(self.c)):
            self.c[i]=d[i]


def sub(a,b,assign_object):
    d=a-b
    assign_object.assign(d)


@tf.function
def f(a,b,assign_object):
    sub(a,b,assign_object)


assign_object=assign()
f(a,b,assign_object)