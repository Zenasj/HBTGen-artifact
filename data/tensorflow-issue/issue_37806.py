import tensorflow as tf

class test(tf.Module):
    pass

a = test()
a.var = tf.Variable(1)
a.const = tf.constant(2)
a.var = tf.constant(a.var.numpy())

tf.saved_model.save(a, "a")
al = tf.saved_model.load("a")

al.const

al.var