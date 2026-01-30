import tensorflow as tf
from tensorflow import keras

input = tf.keras.Input(shape=(100,1), name='input')

x_0 = Conv1D(20,1,strides=1,activation='relu', name='C0')(input)

x_1 = Conv1D(50,2,strides=1, activation='relu', name='C1')(x_0)
x_1 = Dropout(0.3, name='DR1')(x_1)
x_1 = GlobalMaxPooling1D(name='MP1')(x_1)

x_2 = Conv1D(35,3,strides=1,activation='relu', name='C2')(x_0)
x_2 = Dropout(0.3, name='DR2')(x_2)
x_2 = GlobalMaxPooling1D(name='MP2')(x_2)

x_3 = Conv1D(25,4,strides=1,activation='relu', name='C3')(x_0)
x_3 = Dropout(0.3, name='DR3')(x_3)
x_3 = GlobalMaxPooling1D(name='MP3')(x_3)

x_4 = Conv1D(20,5,strides=1,activation='relu', name='C4')(x_0)
x_4 = Dropout(0.3, name='DR4')(x_4)
x_4 = GlobalMaxPooling1D(name='MP4')(x_4)

concat = Concatenate(axis=1, name='concat')([x_1, x_2, x_3, x_4])
concat = BatchNormalization(name='BN')(concat)

output = Dense(units=1, activation='linear',name='output')(concat)

model = tf.keras.Model(inputs=input, outputs=output, name='VRP')