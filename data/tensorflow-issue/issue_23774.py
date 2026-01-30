from tensorflow.keras import layers

import tensorflow.keras.layers as KL
from functools import wraps

def decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

class MyLayer(KL.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @decorator
    def call(self, inputs):
        pass
a=MyLayer()