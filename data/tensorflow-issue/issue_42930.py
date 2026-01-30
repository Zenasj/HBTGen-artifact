import sys
for v in sys.modules.values():
    print(v)

import sys
import tensorflow as tf
for v in sys.modules.values():
    print(v)

import unittest

from my_module import MyCoolThing  # this imports tensorflow

class TestMyAwesomeModel(unittest.TestCase):
    def test_cool_thing_warns(self):
        my_cool_thing = MyCoolThing()
        with self.assertWarns(UserWarning):
            my_cool_thing.do_somthing_naughty()