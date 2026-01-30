# This piece of code demonstrates a bug in tf.strings.lower().

import tensorflow as tf

# Text in Slovak language. (It is first half of a panagram.)
# The text is in 3 different casings: Upper case, lower case, and
# mixed case. In the mixed case, all letters with diacritics 
# are upper case and standard English ASCII letters are lower case.
UPPER_CASE = u"STARÝ KÔŇ NA HŔBE KNÍH ŽUJE TÍŠKO POVÄDNUTÉ RUŽE"
LOWER_CASE = u"starý kôň na hŕbe kníh žuje tíško povädnuté ruže"
MIXED_CASE = u"starÝ kÔŇ na hŔbe knÍh Žuje tÍŠko povÄdnutÉ ruŽe"


class TestLowerCase(tf.test.TestCase):

    # This test case passes.
    # It verifies that, in Python, all three texts
    # are the same after lower casing.
    def test_python(self):
        self.assertEqual(UPPER_CASE.lower(), LOWER_CASE)
        self.assertEqual(MIXED_CASE.lower(), LOWER_CASE)
        self.assertEqual(LOWER_CASE.lower(), LOWER_CASE)

    # This test case passes.
    # It demonstrates the current tensorflow behaviour,
    # which is arguably incorrect.
    def test_tensorflow_current(self):
        tf_upper_case = tf.constant(UPPER_CASE, dtype=tf.string)
        tf_mixed_case = tf.constant(MIXED_CASE, dtype=tf.string)
        self.assertAllEqual(tf.strings.lower(tf_upper_case), tf_mixed_case)

    # This test case fails!
    # It demonstrates the desired/expected behavior.
    def test_tensorflow_desired(self):
        tf_upper_case = tf.constant(UPPER_CASE, dtype=tf.string)
        tf_lower_case = tf.constant(LOWER_CASE, dtype=tf.string)
        self.assertAllEqual(tf.strings.lower(tf_upper_case), tf_lower_case)


# Run all three test cases. The first two pass. The third one fails.
TestLowerCase().test_python()
TestLowerCase().test_tensorflow_current()
TestLowerCase().test_tensorflow_desired()