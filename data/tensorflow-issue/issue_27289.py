import tensorflow as tf


def foo():                                                                                                                                                                                                                                                            
    result = tf.constant(0)                                                                                                                                                                                                                                           
    max_val = tf.constant(1.0)                                                                                                                                                                                                                                        
    max_assert = tf.Assert(tf.greater(max_val, 1.01), [max_val])                                                                                                                                                                                                      
    with tf.control_dependencies([max_assert]):                                                                                                                                                                                                                       
        result = tf.identity(result)                                                                                                                                                                                                                                  
    return result                                                                                                                                                                                                                                                     


class DummyTestCase(tf.test.TestCase):                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                      
    def testRaises(self):                                                                                                                                                                                                                                             
        with self.test_session() as sess:                                                                                                                                                                                                                             
            with self.assertRaises(tf.errors.InvalidArgumentError):                                                                                                                                                                                                   
                sess.run(foo())                                                                                                                                                                                                                                       


if __name__ == '__main__':                                                                                                                                                                                                                                            
    tf.test.main()