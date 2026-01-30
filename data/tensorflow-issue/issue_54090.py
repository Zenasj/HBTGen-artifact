from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

def createModel(look_back=1):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # fixed learning rate
    # lr = 1e-2

    # learning rate schedule
    lr = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5,
                                               decay_steps=10000,
                                               decay_rate=0.99)

    # opt = optimizers.SGD(learning_rate=lr)
    # opt = optimizers.SGD(learning_rate=lr, momentum=0.8, nesterov=False)
    # opt = optimizers.RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-08)
    opt = optimizers.Adam(learning_rate=lr)
    # opt = optimizers.Adadelta(learning_rate=lr)
    # opt = optimizers.Adagrad(learning_rate=lr)
    # opt = optimizers.Adamax(learning_rate=lr)
    # opt = optimizers.Nadam(learning_rate=lr)
    # opt = optimizers.Ftrl(learning_rate=lr)

    model.compile(optimizer=opt, loss='mse')
    # model.summary()
    return model