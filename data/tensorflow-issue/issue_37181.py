import tensorflow as tf

test_input = tf.ones((1000, 1), dtype=tf.int64)
label = tf.ones((1000, 1))
m_input = Input(shape=(1,))
dense = Embedding(5000000, 16,name='embed')(m_input)
dense = Dense(10)(dense)
dense = Dense(5)(dense)
output = Dense(1,'sigmoid')(Flatten()(dense))
model = Model(inputs=[m_input], outputs=[output])
model.compile("adam", loss='binary_crossentropy')
model.fit(test_input, label, steps_per_epoch=1000, epochs=1)