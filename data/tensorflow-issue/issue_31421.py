from tensorflow import keras
from tensorflow.keras import optimizers

# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})

# TensorFlow 2.0
outputs = f(input)

# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)

# Example
import os
import pickle

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


# Create a simple model
inputs = tf.keras.Input(shape=(784,), name='img')

x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

# Compile
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)

# Save to SavedModel 
output_dir = "/tmp/workdir/v1/"
output_directory = Path(output_dir)
output_directory.mkdir(parents=True, exist_ok=True)

model_save_path = os.path.join(output_dir,'model')

model.save(tf_model_save_path, save_format="tf")


# Load the SavedModel

saved_model = tf.saved_model.load(model_save_path, tags=['serve'])

# Get the 'predict' concrete function
infer = saved_model.signatures['serving_default']

pickle_file = os.path.join(output_dir, 'serialized')

# Try serialize the function 
with open(pickle_file, 'wb') as f:
    pickle.dump(infer, f)

# Expect error 
# _pickle.PickleError: can't pickle repeated message fields, convert to list first

# Get the 'predict' concrete function
infer = saved_model.signatures['serving_default']

outputs = session.run(f(placeholder), feed_dict={placeholder: input})

import tensorflow as tf

from pyspark import SparkFiles
from pyspark.sql.functions import udf
import pyspark.sql.types as T
from pyspark.sql import Row

print(tf.__version__)

S3_PREFIX = "s3://"

MODEL_BUCKET = "my-models-bucket"
MODEL_PATH = "path/to/my/model/dir"
MODEL_NAME = "model"

S3_MODEL = f"{S3_PREFIX}{MODEL_BUCKET}/{MODEL_PATH}/{MODEL_NAME}"

print("Fetching model", S3_MODEL)

# Add model to all workers
spark.sparkContext.addFile(S3_MODEL, recursive=True)

# In this example, the SavedModel has the following format:

# inputs = tf.keras.Input(shape=(784,), name='img')
# x = layers.Dense(64, activation='relu')(inputs)
# x = layers.Dense(64, activation='relu')(x)
# outputs = layers.Dense(10, activation='softmax')(x)
# model = tf.keras.Model(inputs=inputs, # outputs=outputs, name='mnist_model')

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255

rows = list(map(lambda n: Row(img=[n.tolist()]), x_test))

schema = T.StructType([T.StructField('img',T.ArrayType(T.ArrayType(T.FloatType())))])

input_df = spark.createDataFrame(rows, schema=schema)

# Simple memoization helper with a single cache key
def compute_once(f):
    K = '0'
    cache = {}
    
    def wrapper(x):
        # Set on first call
        if K not in cache:
            cache[K] = f(x)
        
        return cache[K]

    return wrapper
    

def load_model(model_name):
    # Models are saved under the SparkFiles root directory
    root_dir = SparkFiles.getRootDirectory()
    export_dir = f"{root_dir}/{model_name}"
    
    return tf.saved_model.load(export_dir, tags=['serve'])
    

# Only load the model once per worker!
# The reduced disk IO makes prediction much faster
memo_model_load = compute_once(load_model)

def get_model_prediction(model_name, input):
    """
    Note: 
        TF session is scoped to where the model is loaded.
        All calls to the model's ConcreteFunciton must be in the same scope as
        the loaded model (i.e in the same function!)
        
        If not, TF will throw errors for undefined/ variables
    """
    # Load the predict function (from disk or cache)
    m = memo_model_load(model_name)
    
    # Save the predict signature
    pred_func = m.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    return pred_func(input)

# Decorator with return type of UDF
@udf("array<array<float>>")
def infer(data):
    # Cast the input to a Tensor
    input_data = tf.constant(data)
    
    # Returns a dict of the form { TENSOR_NAME: Tensor }
    outputs = get_model_prediction(MODEL_NAME, input_data)

    # Assuming we have a single output
    output_tensor = list(outputs.values())[0]
    
    # Convert back to regular python
    output_value = output_tensor.numpy().tolist()
    
    return output_value

predictions_df = input_df.withColumn("predictions", infer("img"))

# All done :) 
predictions_df.show(vertical=True)