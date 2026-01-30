def getKerasModel(ndim):

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(ndim,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

# pipeline is doing scales and one hot encoding
X_train2 = full_pipeline.fit_transform(X_train)
X_val2 = full_pipeline.transform(X_val)

model = getKerasModel(ndim=X_train2.shape[1])
model.fit(X_train2, y_train,epochs=5, batch_size=32, verbose=True, validation_data = (X_val2, y_val))