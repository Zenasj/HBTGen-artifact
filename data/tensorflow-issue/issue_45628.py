import random
from tensorflow.keras import layers

from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split

RANDOM_SEED = 40
tf.random.set_seed(RANDOM_SEED)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

values = df.values.astype('float32')
n_steps = 5
X, y = split_sequence(values, n_steps)

model = Sequential()
model.add(LSTM(200, activation='relu',  input_shape=(n_steps,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(
50, activation='relu'))
model.add(Dense(1))

MSE  = metrics.mean_squared_error(y_test,y_pred)
RMSE = sqrt(metrics.mean_squared_error(y_test,y_pred))
MAE  = metrics.mean_absolute_error(y_test,y_pred)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (MSE, RMSE,MAE))

with open("MSE.txt", "w") as text_file:
        MSE=str(MSE)
        text_file.write(MSE)
with open("RMSE.txt", "w") as text_file:
        RMSE=str(RMSE)
        text_file.write(RMSE)
with open("MAE.txt", "w") as text_file:
        MAE=str(MAE)
        text_file.write(MAE)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

from hashlib import md5
f = open("MSE.txt", "r")
s=f.read()
s=float(s)
s=round(s,3)
f1=open("RMSE.txt","r")
s1=f1.read()
s1=float(s1)
s1=round(s1,3)
f2=open("MAE.txt","r")
s2=f2.read()
s2=float(s2)
s2=round(s2,3)
if (md5(str(s).encode()).hexdigest() == '51ad543f7ac467cb8b518f1a04cc06af') and (md5(str(s1).encode()).hexdigest() == '6ad48a76bec847ede2ad2c328978bcfa') and (md5(str(s2).encode()).hexdigest() == '64bd1e146726e9f8622756173ab27831'):

	print("Your MSE,RMSE and MAE Scores matched the expected output")
else :
	print("Your MSE,RMSE and MAE Scores does not match the expected output")