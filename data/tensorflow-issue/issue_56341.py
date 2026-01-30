import math
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_model():
    x1 = linspace(0,4*pi,100000)
    x2 = linspace(2,20,100000)
    y = cos(x1) + x2/5

    model = Sequential()
    model.add(Dense(100, activation='elu', input_shape=(2,)))  # elu used so that smooth, continuous functions are differentiated
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    x_in = np.vstack((x1,x2)).T
    y_out = y

    x_train, x_valid, y_train, y_valid = train_test_split(x_in, y_out, test_size=0.3, shuffle= True)

    epochs = 100
    batch_size = 110

    h0 = model.fit(x_train, y_train, validation_data=(x_valid,y_valid), batch_size=batch_size,epochs=epochs)

    return x_in, y_out, model

def analyze(model, x_in): 

    grado1 = np.zeros((100000,2))
    grado2 = np.zeros((100000,2))

    for i in range(0,100000):
        x = tf.Variable([[x_in[i,0],x_in[i,1]]])

        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as t:
            t.watch(x)  
            out1 = model(x)
            out2 = tf.math.cos(x[:,0]) + x[:,1]/5  # this works using both a Variable array as well as separated taped variables, i.e., x1t, x2t
        gradients1 = t.gradient(out1, x)
        gradients2 = t.gradient(out2, x)
        if i % 100 == 0:
            print(i)
        grado1[i,:] = gradients1.numpy()
        grado2[i,:] = gradients2.numpy()

    ymodel = model.predict([x_in])

    return grado1, grado2, ymodel


print('x_in, y_out, model = make_model()')
print('grado1, grado2, ymodel = analyze(model, x_in)')

def main(x_in, y_out, grado1, grado2, ymodel):

    # project the gradients onto the line x1 = x2
    direction = np.array([4 * pi, 20 - 2])
    direction /= np.linalg.norm(direction)

    proj_grado1 = np.matmul(grado1, direction)
    proj_grado2 = np.matmul(grado2, direction)

    plt.figure(figsize=(15,4))
    ax = plt.subplot(111)
    plt.ylabel('dy / d_direction')
    plt.xlabel('x_1')
    plt.legend(loc='lower left')

    # ax2 = ax.twinx()
    # ax2.plot(x_in[:,i],grado1[:,i],'b--',label='grad1_x'+str(i+1)+' model')
    # ax2.plot(x_in[:,i],grado2[:,i],'g',label='grad2_x'+str(i+1)+' autodiff formula')

    ax.plot(x_in[:,0],proj_grado1[:],'b--',label='line_grad1_x model')
    ax.plot(x_in[:,0],proj_grado2[:],'g',label='line_grad2_x autodiff formula')
    # if i == 0:
        # ax2.plot(x_in[:,0],-sin(x_in[:,0]),'r--',label='grad2_x'+str(i+1)+' explicit formula')
    # elif i == 1:
        # ax2.plot(x_in[:,1],(1/5)*ones(len(x_in[:,1])),'r--',label='grad2_x'+str(i+1)+' explicit formula')
    # plt.ylabel('dy/dx_1')
    # plt.xlabel('x_1')

    plt.legend()
    plt.show()