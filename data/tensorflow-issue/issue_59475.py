import tensorflow as tf
import numpy as np


class test:
    def __init__(self):
        self.counter=None
    
    
    def sum_func(self):
        if np.sum(self.counter)==1875:
            return True
    
    
    @tf.function
    def tf_func(self):
        flag=self.sum_func()
        return flag
    
    
    def counter_func(self):
        while True:
            for i in range(1875):
                if i==0:
                    self.counter=np.array(1)
                else:
                    self.counter=np.append(self.counter,np.array(1))
                flag=self.tf_func()
                if flag==True:
                    return

t=test()
t.counter_func()