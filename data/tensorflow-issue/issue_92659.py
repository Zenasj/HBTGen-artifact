import random

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import voxelmorph as vxm

tf.random.set_seed(seed=0)
print(tf.__version__)
from tensorflow import keras

model_path = "/home/users/giuseppe.sorrentino/SODA/models/abdomreg_intra.h5"
model = vxm.networks.VxmDense.load(
    "/home/users/giuseppe.sorrentino/SODA/models/abdomreg_intra.h5"
)

save_path = os.path.join(os.getcwd(), "model/simple/")
tf.saved_model.save(model, save_path) 

@tf.function
def infer(moving, fixed):
    return model([moving, fixed])

inp0, inp1 = model.inputs

concrete_func = infer.get_concrete_function(
    moving=tf.TensorSpec(shape=inp0.shape, dtype=inp0.dtype, name=inp0.name.split(':')[0]),
    fixed =tf.TensorSpec(shape=inp1.shape, dtype=inp1.dtype, name=inp1.name.split(':')[0])
)

frozen_func = convert_variables_to_constants_v2(concrete_func)
tf.io.write_graph(
    graph_or_graph_def=frozen_func.graph,
    logdir=os.getcwd(),
    name="output/frozen_graph.pbtxt",
    as_text=True
)