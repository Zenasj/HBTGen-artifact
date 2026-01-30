import tensorflow as tf
from tensorflow import keras

def ctc_batch_cost(args):
    y_pred,y_true = args
    input_length = y_true[:,1:2]+5
    label_length = y_true[:,1:2]+5
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
  
inp = tf.keras.Input(shape=(28*28,))
x = l.Reshape(input_shape=(28*28,), target_shape=(28, 28))(inp)
x = l.Dense(10, activation='softmax')(x)
loss = l.Lambda(ctc_batch_cost, output_shape=(1,), name='ctc')([x, inp ])

model = tf.keras.Model(inputs=[inp], outputs=loss)

model.compile(optimizer='adam', 
              loss={'ctc': lambda y_true, y_pred: y_pred},
              metrics=['accuracy'])