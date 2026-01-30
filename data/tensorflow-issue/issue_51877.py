import numpy as np
import unittest
class test(unittest.TestCase):
    def test_assert_1(self):
        actual = np.array([0.93933064])
        expected = 0.9393305437365417
        self.assertAlmostEqual(expected, actual, 2)

    def test_assert_2(self):
        actual = np.array([0.93933064])
        expected = 0.9393305437365417
        self.assertEqual(list(actual.shape), [1])
        self.assertAlmostEqual(expected, actual[0], 2)

f = test()
f.test_assert_2()
f.test_assert_1()

import numpy as np
import unittest
class test(unittest.TestCase):
    def test_assert_1(self):
        actual = np.array([0.93933064])
        expected = 0.9393305437365417
        self.assertAlmostEqual(expected, actual, 2)

    def test_assert_2(self):
        actual = np.array([0.93933064])
        expected = 0.9393305437365417
        self.assertEqual(list(actual.shape), [1])
        self.assertAlmostEqual(expected, actual[0], 2)

f = test()
f.test_assert_2()
f.test_assert_1()

import numpy as np
class A(np.ndarray):

  def __round__(self, places):
    return np.round_(self, places)

import unittest
class test(unittest.TestCase):
    def test_assert_1(self):
        actual = np.array([0.93933064]).view(type=A)
        expected = 0.9393305437365417
        self.assertAlmostEqual(expected, actual, 2)
    def test_assert_2(self):
        actual = np.array(0.93933064).view(type=A)
        expected = 0.9393305437365417
        self.assertAlmostEqual(expected, actual, 2)
    def test_assert_3(self):
        actual = np.array(0.93933064).view(type=np.ndarray)
        expected = 0.9393305437365417
        self.assertAlmostEqual(expected, actual, 2)

f = test()
f.test_assert_1() 
f.test_assert_2()
f.test_assert_3()