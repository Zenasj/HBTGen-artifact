import tensorflow as tf

class A():
    def __init__(self):
        self.lst = [1, 2, 3]
        self.sess = tf.Session()
        self.total_length = tf.constant(len(self.lst))

    def loop(self, i):
        pr = tf.print(i)
        current_value = self.lst[i.eval(session=self.sess)]
        with tf.control_dependencies([pr]):
            i = tf.add(i, 1)
        return [i]

    def cond(self, i):
        return tf.less(i, self.total_length) 

    def run(self):
        i = tf.constant(0)
        while_op = tf.while_loop(self.cond, self.loop, [i])
        final_i = self.sess.run(while_op)


if __name__ == "__main__":
    obj = A()
    obj.run()

import tensorflow as tf

class A():
    def __init__(self):
        self.lst = tf.constant([1, 2, 3])
        self.sess = tf.Session()
        self.total_length = tf.size(self.lst)

    def loop(self, i):
        pr = tf.print(i)
        current_value = self.lst[i]
        with tf.control_dependencies([pr]):
            i = tf.add(i, 1)
        return [i]

    def cond(self, i):
        return tf.less(i, self.total_length) 

    def run(self):
        i = tf.constant(0)
        while_op = tf.while_loop(self.cond, self.loop, [i])
        final_i = self.sess.run(while_op)


if __name__ == "__main__":
    obj = A()
    obj.run()

class A():
    def __init__(self):
        self.lst = [1, 2, 3]
        self.sess = tf.Session()
        self.total_length = tf.size(self.lst)

    def loop(self, i):
        pr = tf.print(i)
        current_value = i.eval()
        pr_2 = tf.print(self.lst[current_value])
        with tf.control_dependencies([pr, pr_2]):
            i = tf.add(i, 1)
        return [i]

    def cond(self, i):
        return tf.less(i, self.total_length) 

    def run(self):
        i = tf.constant(0)
        while_op = tf.while_loop(self.cond, self.loop, [i])
        final_i = self.sess.run(while_op)


if __name__ == "__main__":
    obj = A()
    obj.run()