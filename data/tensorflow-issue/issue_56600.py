import numpy as np 
import pandas as pd 
import tracemalloc
tracemalloc.start()
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_labels = train.iloc[:,-3:]
features = pd.concat([train.iloc[:,1:-3], test.iloc[:,1:]])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features =scaler.fit_transform(features)
scaled_train = scaled_features[:train.shape[0]]
scaled_test = scaled_features[train.shape[0]:]
print(scaled_train.shape, scaled_test.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_train, train_labels, random_state = 42, test_size = 0.3)
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[8]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(3),
])
model.compile( optimizer = "adam", loss = "mae")
history = model.fit( x_train, y_train, validation_data=(x_test, y_test), batch_size = 256, epochs = 50, verbose = False)
hstry_df = pd.DataFrame(history.history)

submissions = model.predict(scaled_test)

submission_df = pd.DataFrame(submissions, columns = train_labels.columns)
submission_df['date_time'] = test['date_time']
submission_df.to_csv("submissions.csv", index=False)


current3, peak3 = tracemalloc.get_traced_memory()
print("Get_dummies memory usage is {",current3 /1024/1024,"}MB; Peak memory was :{",peak3 / 1024/1024,"}MB")