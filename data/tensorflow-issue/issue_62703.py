from tensorflow import keras
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
# from tensorflow.keras import mixed_precision
from keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# 加载数据
data_folder = "RamanLib.npy"
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

for mineral_class in os.listdir(data_folder):
    class_folder = os.path.join(data_folder, mineral_class)
    for fold in range(5):  # 假设你有5折数据
        X_train = np.load(os.path.join(class_folder, f"fold{fold}_train_data.npy"))
        X_test = np.load(os.path.join(class_folder, f"fold{fold}_test_data.npy"))

        # 增加一个维度以满足RNN的输入要求
        X_train = X_train[:, :, np.newaxis]
        X_test = X_test[:, :, np.newaxis]

        y_train = [mineral_class] * len(X_train)
        y_test = [mineral_class] * len(X_test)

        X_train_list.append(X_train)
        y_train_list.extend(y_train)
        X_test_list.append(X_test)
        y_test_list.extend(y_test)

X_train = np.vstack(X_train_list)  # 将列表转换为numpy数组
y_train = np.array(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.array(y_test_list)

# 对标签进行编码
encoder = LabelEncoder()
encoder.fit(np.concatenate((y_train, y_test)))
y_train = np_utils.to_categorical(encoder.transform(y_train))
y_test = np_utils.to_categorical(encoder.transform(y_test))

def create_model(neurons=128, l2_weight=0.001, dropout_rate=0, learning_rate=0.001):
    model = models.Sequential()
    
    model.add(layers.Bidirectional(layers.LSTM(neurons, input_shape=X_train.shape[1:], return_sequences=True,
                               kernel_regularizer=l2(l2_weight))))
    model.add(layers.Dropout(dropout_rate))
    
    
    model.add(layers.Bidirectional(layers.LSTM(neurons, return_sequences=True, kernel_regularizer=l2(l2_weight))))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Bidirectional(layers.LSTM(neurons, return_sequences=True, kernel_regularizer=l2(l2_weight))))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Bidirectional(layers.LSTM(neurons, kernel_regularizer=l2(l2_weight))))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(y_train.shape[1]))
    model.add(layers.Activation('softmax'))


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 创建并编译模型
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model(learning_rate=0.001)

# 训练模型
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32)

# 评估模型
train_loss, train_acc = model.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_acc}")#训练完成后，使用测试集进行评价

test_loss, test_acc = model.evaluate(X_test, y_test)#查看模型在训练集上的损失和精度值
print(f"Test Accuracy: {test_acc}")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)