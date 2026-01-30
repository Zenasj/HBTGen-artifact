from tensorflow.keras import layers

py
import tensorflow as tf
print(tf.__version__)
import numpy as np
import os
from tensorflow import keras
from tensorflow.python.lib.io import file_io

# os.environ['AWS_ACCESS_KEY_ID'] = "Q3AM3UQ867SPQQA43P2F"
# os.environ['AWS_SECRET_ACCESS_KEY'] = "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
# os.environ['AWS_REGION'] = "us-east-1"
# os.environ['S3_ENDPOINT'] = "play.min.io"
# os.environ['S3_USE_HTTPS'] = "1"
# os.environ['S3_VERIFY_SSL'] = "1"

os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_REGION'] = "us-east-1"
os.environ['S3_ENDPOINT'] = "localhost:9000"
os.environ['S3_USE_HTTPS'] = "0"
os.environ['S3_VERIFY_SSL'] = "0"

print(file_io.stat('s3://tfbuck6'))

def get_model():
  # Create a simple model.
  inputs = keras.Input(shape=(32,))
  outputs = keras.layers.Dense(1)(inputs)
  model = keras.Model(inputs, outputs)
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

tfmodel = get_model()
keras.experimental.export_saved_model(tfmodel, "s3://tfbuck6", serving_only=True)