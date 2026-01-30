import tensorflow as tf

class C(object):
    def f(self):
        # error disappear if \ in the following line is removed
        a = \
            1
        return a

obj = C()

@tf.function
def func():
    mem =  obj.f()
    return mem

def main():
    print(func())

if __name__ == "__main__":
    main()