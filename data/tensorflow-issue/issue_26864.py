import tensorflow as tf

class A(tf.Module):
    @tf.function
    def func(self, x: int):
        pass


a = A() 
tf.saved_model.save(a, export_dir=".")

def func(self, x: "int"):
    ...

from __future__ import annotations