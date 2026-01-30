from tensorflow.keras import layers

m1 = Sequential([
    Input((100,)),
    Dense(20),
    Dense(10),
])

m2 = Sequential([
    Input((100,)),
    m1,
    Dense(5),
])

test1 = Model(m1.input, m1.layers[1].output)
test2 = Model(m2.input, m2.layers[1].output)

test3 = Model(m2.input, m2.layers[0].layers[1].output)

test4 = Model(m2.layers[0].input, m2.layers[0].layers[1].output)

x = Input((128,))
y = Dense(64)(x)
m1 = Model(x, y)

m2 = Sequential([Input((128,1)), Dense(64)])

x = Input((128,))
y = Dense(64)(x)
m1 = Model(x, y)

m2 = Sequential([Input((128,)), Dense(64)])

m3 = Sequential([InputLayer((128,)), Dense(64)])

m1 = Sequential([
    Input((100,)),
    Dense(20),
    Dense(10),
])

m2 = Sequential([
    Input((100,)),
    m1,
    Dense(5),
])

test3 = Model(m2.input, m2.layers[0].layers[1].output)

from tensorflow import keras
m1 = keras.Sequential([
    keras.layers.Input((100,),name="m1_input"),
    keras.layers.Dense(20),
    keras.layers.Dense(10),
])

m2 = keras.Sequential([
    keras.layers.Input((100,),name="m2_input"),
    m1,
    keras.layers.Dense(5,name="bni"),
])
test1 = keras.Model(m1.input, m1.layers[1].output)
test2 = keras.Model(m2.get_layer(index=0).input, m2.get_layer(index=0).output)
test3 = keras.Model(m2.get_layer(index=0).input, outputs=m2.layers[0].layers[1].output)
test4 = keras.Model(m2.layers[0].input, m2.layers[0].layers[1].output)

print(m2.get_layer(index=0).input.name)
print(m2.input.name)
test3 = keras.Model(m2.get_layer(index=0).input,m2.layers[0].layers[1].output)

test4 = Model(m2.layers[0].input, m2.layers[0].layers[1].output)