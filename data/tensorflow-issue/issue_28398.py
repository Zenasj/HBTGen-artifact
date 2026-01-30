from tensorflow.keras import layers
from tensorflow.keras import optimizers

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.python import keras

iris = datasets.load_iris()

scl = StandardScaler()
ohe = OneHotEncoder(categories='auto')
data_norm = scl.fit_transform(iris.data)
data_target = ohe.fit_transform(iris.target.reshape(-1,1)).toarray()
train_data, val_data, train_target, val_target = train_test_split(data_norm, data_target, test_size=0.1)
train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2)


train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_target))
train_dataset = train_dataset.batch(32).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_target))
test_dataset = test_dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_target))
val_dataset = val_dataset.batch(12).repeat()

mdl = keras.Sequential([
    keras.layers.Dense(16, input_dim=4, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')]
)

mdl.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.categorical_crossentropy,
    metrics=[keras.metrics.categorical_accuracy]
    )

history = mdl.fit(train_dataset, epochs=10, steps_per_epoch=15, validation_data=val_dataset, validation_steps=12)
results = mdl.evaluate(test_dataset, steps=15)
comparison = mdl.predict_classes(test_dataset)