import dill
import pickle

class B:
    def __init__(self, function):
        self.function = function

    def __getstate__(self):
        print('serializing ', self)
        r = pickle.dumps(self.function)
        print(r)
        return r

class A:
    def f_a(self, a):
        return a + 1

    def __init__(self, x):
        self.z = B(self.f_a)    
        self.val = x

a = A(3)
d = pickle.dumps(a)