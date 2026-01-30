def my_graph(use_identity_op) :
   """Dummy function for testing
   
   :param use_identity_op: whether to apply the identity operator to the 
   a variable or not
   :type use_identity_op: bool
   :return: a dummy vector
   :rtype: list(float)
   """
   tf.compat.v1.reset_default_graph()
   tf.compat.v1.set_random_seed(32)
   a = tf.compat.v1.get_variable('my_a', [5], initializer=None)
   if use_identity_op : 
        a = tf.identity(a)
   b = tf.compat.v1.get_variable('my_b', [5], initializer=None)
   c = a + b
   session = tf.compat.v1.Session()
   session.run(tf.compat.v1.global_variables_initializer())
   return session.run(c).tolist()

import tensorflow as tf
import unittest

def my_graph(use_identity_op) :
   """Dummy function for testing
   
   :param use_identity_op: whether to apply the identity operator to the 
   a variable or not
   :type use_identity_op: bool
   :return: a dummy vector
   :rtype: list(float)
   """
   tf.compat.v1.reset_default_graph()
   tf.compat.v1.set_random_seed(32)
   a = tf.compat.v1.get_variable('my_a', [5], initializer=None)
   if use_identity_op : 
        a = tf.identity(a)
   b = tf.compat.v1.get_variable('my_b', [5], initializer=None)
   c = a + b
   session = tf.compat.v1.Session()
   session.run(tf.compat.v1.global_variables_initializer())
   return session.run(c).tolist()

class TestTFIdentity(unittest.TestCase): 
   def test_my_graph_is_deterministic_CPU(self):
      
      with tf.device('/device:CPU:0'):
         #The 3 calls below should be equivalent.

         run1 = my_graph(use_identity_op=False)
         run2 = my_graph(use_identity_op=False)
         self.assertEqual(run1, run2)
         
         run3 = my_graph(use_identity_op=True)
         #This fails though I feel it shouldn't 
         self.assertEqual(run1, run3)

if __name__ == '__main__':
    unittest.main()