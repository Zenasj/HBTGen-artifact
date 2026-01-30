from tensorflow.keras import optimizers

import urllib.request as request
import tensorflow as tf
import pandas as pd


def download_data(download_path: str):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    header_line = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n'

    # Download the data and add the column headers
    request.urlretrieve(url, 'heart.data')
    with open(download_path, 'w') as output:
        output.write(header_line)
        with open('heart.data', 'r') as input_data:
            output.writelines(input_data.readlines())


def preprocess_df(df, categorical_columns):
    """Ensure categorical columns are treated as string inputs"""
    col_types = {key : str for key in categorical_columns.keys()}
    df = df.astype(col_types)
    return df


def df_to_dataset(dataframe, target_column='target', shuffle=True, batch_size=5):
    """Dataset preparation code from the tensorflow tutorial"""
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_column)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), tf.one_hot(labels, depth=2)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


if __name__ == '__main__':

    # Download dataset
    data_path = 'heart.csv'
    download_data(data_path)
    df = pd.read_csv(data_path)

    # Setup feature columns
    numeric_columns = ["age", "chol"]
    categorical_columns = {"thal": df['thal'].unique()}
    feature_columns = {}
    inputs = {}
    for feature_name in numeric_columns:
        feature_columns[feature_name] = tf.feature_column.numeric_column(feature_name)
        inputs[feature_name] = tf.keras.Input(name=feature_name, shape=(), dtype=tf.float32)

    for feature_name, vocab in categorical_columns.items():
        vocab.sort()
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
        feature_columns[feature_name] = tf.feature_column.indicator_column(cat_col)
        inputs[feature_name] = tf.keras.Input(name=feature_name, shape=(), dtype=tf.string)

    # Prepare input data
    df = preprocess_df(df, categorical_columns)
    batch_size = 5  # A small batch sized is used for demonstration purposes
    train_ds = df_to_dataset(df, target_column='target', batch_size=batch_size)

    # Create Model
    input_tensors = []
    feature_names = list(feature_columns.keys())
    feature_names.sort()
    for column_name in feature_names:
        features = feature_columns[column_name]
        x = tf.keras.layers.DenseFeatures(features, name=f'{column_name}_feature')(inputs)
        input_tensors.append(x)

    x = tf.keras.layers.Concatenate()(input_tensors)
    x = tf.keras.layers.Dense(units=24, activation='relu', name='dense_0')(x)
    x = tf.keras.layers.Dense(units=24, activation='relu', name='dense_1')(x)
    y_pred = tf.keras.layers.Dense(units=2, activation='softmax', name='output_layer')(x)
    model = tf.keras.Model(inputs=inputs, outputs=y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'],
                  run_eagerly=True)

    model.summary()
    model.fit(train_ds, epochs=1)

    # Create a new keras model to extract the features
    # the actual model is using.
    outputs = []
    for column_name in feature_names:
        outputs.append(model.get_layer(f'{column_name}_feature').output)

    feature_extractor = tf.keras.Model(model.input, outputs)

    for i, (X, _) in enumerate(train_ds):

        # Predict works as it calls keras.engine.training_utils.standardize_input_data() internally
        # this modifies the input so that if extra columns are passed they are removed and column
        # order is changed as per the model inputs specified.
        # out = feature_extractor.predict(X)

        # Model call() doesn't use the above util method and thus fails
        # as the ordering of the input columns doesn't match the input
        out = feature_extractor(X)
        print(out)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print("tensorflow version",tf.__version__)

# make a null base model
baseinputs = keras.Input(shape=(1,), name='base_input')
base = keras.Model(inputs=baseinputs, outputs=baseinputs)

# Use it to construct a bigger model.
# (I now know there is a better way to make this model.)
Nin,Nout,Nparallel = 1,1,2
parallel_inputs = keras.Input(shape=(Nin,Nparallel), name='parallel_input0')
# apply base NN to each parallel slice; each outputs (?,Nout)
xs = [base(layers.Lambda(lambda x : x[:,:,i],name='view'+str(i))(parallel_inputs)) for i in range(Nparallel)]
# reshape each of them to (?,Nout,1)
xs = [layers.Reshape((Nout,1))(x) for x in xs]
# concatenate on the third direction to get (?,Nout,Nparallel)
cx = layers.Concatenate()(xs)
# create input scalars for weighted sun (?,Nparallel)
weight_inputs = keras.Input(shape=(Nparallel,), name='parallelScalars')
# do a weighted sum over the third direction to get (?,Nout)
out = layers.Dot((-1,-1))([cx,weight_inputs])
wrapper = keras.Model(inputs=[weight_inputs,parallel_inputs], outputs=out, name='parallelwrapper')

# Make tiny example and try predict and call
w = np.array([[7.,11.]],dtype='float32')
v = np.array([[[3.,5.]]],dtype='float32')
wrappercall = wrapper([w,v]).numpy()
wrapperpredict = wrapper.predict([w,v])
print("wrapper ||call - predict|| =",np.linalg.norm(wrappercall-wrapperpredict))
print("wrapper predict =",wrapperpredict,"; correct is 3*7+5*11=",3*7+5*11)
print("wrapper call =",wrappercall,"; appears to do 5*(7+11)=",5*(7+11))

"""
Outputs:

/usr/lib/python3/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
tensorflow version 2.0.0
2019-10-21 14:27:01.188059: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-21 14:27:01.211364: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 1800000000 Hz
2019-10-21 14:27:01.211915: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x54e1d50 executing computations on platform Host. Devices:
2019-10-21 14:27:01.211940: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
wrapper ||call - predict|| = 14.0
wrapper predict = [[76.]] ; correct is 3*7+5*11= 76
wrapper call = [[90.]] ; appears to do 5*(7+11)= 90


"""

tf.__version__