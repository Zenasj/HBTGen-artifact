import tensorflow as tf
from tensorflow.keras import layers

inn = tf.keras.Input((1,),name="in1")
d1 = tf.keras.layers.Dense(1, name="d1")(inn)
d2 = tf.keras.layers.Dense(1, name="d2")(inn)
m1 = tf.keras.Model(inputs=inn, outputs=[d1,d2], name="model1")
in2_1 = tf.keras.Input((1,),name="in2_1")
in2_2 = tf.keras.Input((1,),name="in2_2")
m2 = tf.keras.Model(inputs=[in2_1,in2_2],outputs=[in2_1 + in2_2],name="model2")

# combined model
in0 = tf.keras.Input((1,),name="in0")
m = tf.keras.Model(inputs=in0, outputs=m2(m1(in0)))
tf.keras.utils.plot_model(m, show_shapes=True, expand_nested=True)

from tensorflow import keras

a = keras.Input(shape=(1,), name="a")
b = keras.Input(shape=(1,), name="b")
c = keras.layers.Concatenate(name="c")([a, b])
inner = keras.Model(inputs=[a, b], outputs=c, name="inner")

x = keras.Input(shape=(1,), name="x")
y = keras.Input(shape=(1,), name="y")
z = inner([x, y])
outer = keras.Model(inputs=[x, y], outputs=z, name="outer")

keras.utils.plot_model(outer, show_shapes=True, expand_nested=True)