from tensorflow import keras

import cloudpickle  # cloudpickle.__version__ == '1.2.1'
import tensorflow as tf  # tf.__version__ == '2.0.0-rc0'

def f():
    tf.keras.Sequential

cloudpickle.loads(cloudpickle.dumps(f))  # This fails.

_p = print
import cloudpickle  # cloudpickle.__version__ == '1.2.1'
import tensorflow as tf  # tf.__version__ == '2.0.0-rc0'

def f():
  _p("f() called")
  tf.keras.Sequential
  _p("f() ending")

_p("Dumping...")
s = cloudpickle.dumps(f)
_p("dumped, loading...")
cloudpickle.loads(s)
_p("done")