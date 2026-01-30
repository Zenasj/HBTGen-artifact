import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("model/train/converted_model.tflite", "wb").write(tflite_quant_model)
print('tflite convert finish')

# python 3.7.6, Tensorflow 2.2.0
import tensorflow as tf
import numpy as np
import data_preprocess

# 超参数
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 200
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300

EPOCHS = 10
OPT = 'rmsprop'
LOSS = 'binary_crossentropy'
MODEL_DIR = './model/train/'
MODEL_FORMAT = '.h5'
filename_str = "{}sts_{}_{}_epochs_{}{}"
# 模型文件
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS,  str(EPOCHS), MODEL_FORMAT)

print('Indexing word vectors.')
embeddings_index = data_preprocess.build_word_embedding()

train_data_file = 'data/train.txt'
s1, s2, score = data_preprocess.read_train_data(train_data_file)
# print('s1=', s1[0])
left, right, texts = data_preprocess.change_for_tokenizer(s1, s2)
# print('left=', left[0])
# print('texts=', texts[:2])
print('Found %s left.' % len(left))
print('Found %s right.' % len(right))
print('Found %s labels.' % len(score))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

seq_left = tokenizer.texts_to_sequences(left)
# print(seq_left)
seq_right = tokenizer.texts_to_sequences(right)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_left = tf.keras.preprocessing.sequence.pad_sequences(seq_left, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
data_right = tf.keras.preprocessing.sequence.pad_sequences(seq_right, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
score = np.array(score)
# print(data_left)

indices = np.arange(data_left.shape[0])
np.random.shuffle(indices)
data_left = data_left[indices]
data_right = data_right[indices]
score = score[indices]

val_index = int(VALIDATION_SPLIT * data_left.shape[0])
input_train_left = data_left[:-val_index]
input_train_right = data_right[:-val_index]
val_left = data_left[-val_index:]
val_right = data_right[-val_index:]
train_score = score[:-val_index]
val_score = score[-val_index:]

print('Preparing embedding matrix.')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('Training model.')
tweet_a = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
tweet_b = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
tweet_input = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))

embedding_layer = tf.keras.layers.Embedding(nb_words + 1,
                                            EMBEDDING_DIM,
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            weights=[embedding_matrix],
                                            trainable=True)(tweet_input)
conv1 = tf.keras.layers.Conv1D(128, 3, activation='tanh')(embedding_layer)
drop1 = tf.keras.layers.Dropout(0.2)(conv1)
max1 = tf.keras.layers.MaxPooling1D(3)(drop1)
# conv2 = tf.keras.layers.Conv1D(128, 3, activation='tanh')(max1)
# drop2 = tf.keras.layers.Dropout(0.2)(conv2)
# max2 = tf.keras.layers.MaxPooling1D(3)(drop2)
out = tf.keras.layers.Flatten()(max1)

model_encode = tf.keras.models.Model(tweet_input, out)
encoded_a = model_encode(tweet_a)
encoded_b = model_encode(tweet_b)
merged = tf.keras.layers.concatenate([encoded_a, encoded_b])
dense1 = tf.keras.layers.Dense(128, activation='relu')(merged)
dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(128, activation='relu')(dense2)
prediction = tf.keras.layers.Dense(1, activation='sigmoid')(dense3)
model = tf.keras.Model(inputs=[tweet_a, tweet_b], outputs=prediction)
model.summary()
model.compile(optimizer=OPT,
              loss=LOSS,
              metrics=['accuracy'])
# 训练模型
model.fit([input_train_left, input_train_right], train_score, epochs=EPOCHS)

# 保存模型
if not tf.io.gfile.exists(MODEL_DIR):
    tf.io.gfile.makedirs(MODEL_DIR)

model.save(MODEL_FILE)
print('Saved trained model at %s ' % MODEL_FILE)

# 模型转换
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.experimental_new_converter = True
tflite_quant_model = converter.convert()
open("model/train/converted_model.tflite", "wb").write(tflite_quant_model)
print('tflite convert finish')

# 预测测试集
loss_and_accuracy = model.evaluate([val_left, val_right], val_score)
print("Test Loss: {}".format(loss_and_accuracy[0]))
print("Test Accuracy: {}%".format(loss_and_accuracy[1]*100))

# 数据预处理
import word2vec
import pandas as pd
import numpy as np
import jieba

# 词向量文件路径
word_vector_file_path = 'data/sgns.weibo.word.txt'


def build_word_embedding():
    embeddings_index = {}
    f = open(word_vector_file_path)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))  # embeddings_index type=dict
    return embeddings_index


def build_word_vectors():
    wv = word2vec.load(word_vector_file_path)
    vocab = wv.vocab
    vectors = wv.vectors
    print('build_word_dic fun vocab len=', len(vocab))
    # print(vocab)
    # print(word_embedding)
    return vocab, vectors


def read_train_data(train_file):
    df_train = pd.read_csv(train_file, sep='\t', usecols=[0, 1, 2], names=['s1', 's2', 'score'],
                           dtype={'s1': object, 's2': object, 'score': object})
    s1 = df_train.s1.values
    s2 = df_train.s2.values
    score = np.asarray(df_train.score.values, dtype=np.float32)
    print('train data len=', len(score))
    return s1, s2, score


def change_for_tokenizer(s1, s2):
    size = len(s1)
    texts = []
    left = []
    right = []
    for i in range(size):
        seg_left = jieba.lcut(s1[i])
        seg_right = jieba.lcut(s2[i])
        text_left = (' '.join(seg_left))  # .encode('utf-8', 'ignore').strip()
        text_right = (' '.join(seg_right))  # .encode('utf-8', 'ignore').strip()

        texts.append(text_left)
        texts.append(text_right)

        left.append(text_left)
        right.append(text_right)
    return left, right, texts