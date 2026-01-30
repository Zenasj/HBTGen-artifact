from tensorflow.keras import layers
from tensorflow.keras import models

3
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
y = to_categorical(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)
model1 = Sequential([
    Dense(512, activation='tanh', input_shape = X_train[0].shape),
    Dense(512//2, activation='tanh'),
    Dense(512//4, activation='tanh'),
    Dense(512//8, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

3
model = Sequential([
Dense(10, input_shape=(19,)),
Dense(1)
])