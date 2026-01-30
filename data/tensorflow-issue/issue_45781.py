import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

reloaded_model = tf.keras.models.load_model(tmpdir, custom_objects={"MyModelSubclass": MyModelSubclass})
assert isinstance(reloaded_model, MyModelSubclass)

reloaded_model = tf.keras.models.load_model(tmpdir, custom_objects={"MyModelSubclass": MyModelSubclass})
assert isinstance(reloaded_model, MyModelSubclass)

class MyModelSubclass(tf.keras.models.Model):
    pass
model = MyModelSubclass()
model.save(tmpdir)
del MyModelSubclass  # delete class definition since user may not have access to it in a new runtime

reloaded_model = tf.keras.models.load_model(tmpdir)
print(type(reloaded_model))

# Change CustomModel to inherit from `Functional`
CustomModel.__bases__ = (tensorflow.python.keras.engine.functional.Functional,)