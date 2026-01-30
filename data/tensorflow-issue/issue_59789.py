import tensorflow as tf

class TestDummy(tf.test.TestCase):

    def test_dummy_confusing_error(self):
        # this test should fail. But it raises the above confusing error message.
        a = tf.constant(1.00002, dtype=tf.float32)
        b = tf.constant(1.00001, dtype=tf.float32)
        self.assertAlmostEqual(a, b)
    
    def test_dummy_confusing_bug(self):
        # this test should pass with places=2. But it still raises the above confusing error message.
        a = tf.constant(1.00002, dtype=tf.float32)
        b = tf.constant(1.00001, dtype=tf.float32)
        self.assertAlmostEqual(a, b, places=2)

    def test_dummy_no_error(self):
        # this test passes 
        a = tf.constant(1.000000002, dtype=tf.float32)
        b = tf.constant(1.000000001, dtype=tf.float32)
        self.assertAlmostEqual(a, b)

if __name__ == "__main__":
    tf.test.main()