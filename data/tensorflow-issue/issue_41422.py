import numpy as np

X.shape = (5000, 12)
y.shape = (5000, 3, 12)

n_input = 7
generator = TimeseriesGenerator(X, y, length=n_input, batch_size=1)

for i in range(5):
    x_, y_ = generator[i]
    print(x_.shape)
    print(y_.shape)

(1, 7, 12)
(1, 3, 12)
(1, 7, 12)
(1, 3, 12)
...

model = Sequential()
model.add(LSTM(4, input_shape=(None, 12)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit_generator(generator, epochs=3).history

y=np.reshape(y,(-1,y.shape[-1]*y.shape[-2]))

n_input = 7
generator = TimeseriesGenerator(X, y, length=n_input, batch_size=1)

model = Sequential()
model.add(LSTM(64, input_shape=(7, 12)))
model.add(Dense(Y.shape[-1]))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(generator, epochs=3).history

model = Sequential()
model.add(LSTM(4, return_sequences=True, input_shape=(n_input, 12)))
model.add(MaxPool1D(2)) # also AvgPool1D is ok
model.add(Dense(12))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
model.fit(generator, epochs=2)