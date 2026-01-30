import numpy as np
import pandas as pd

libmodel = np.ctypeslib.load_library('libmodel', '/folder/org_tensorflow/')

libmodel.run.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(1,5), flags=('c', 'a')),
np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(1,5), flags=('c', 'a', 'w')),
np.ctypeslib.ctypes.c_float,
np.ctypeslib.ctypes.c_float]

df = pd.read_csv('/folder/trial.csv', header = 0, index_col = None)
feats = pd.read_table('/folder/feature_list.txt', header=None, index_col = None)
feats = feats.iloc[:,0].tolist()
df = df.reindex(columns=feats, fill_value=0)

x = np.require(df.iloc[0,:], np.float32, ('c', 'a'))
y = np.require(np.zeros((1,1)), np.float32, ('c', 'a', 'w'))

x = x.reshape((1,5))
libmodel.run(x, y, x.size, y.size)

load('@org_tensorflow//tensorflow/compiler/aot:tfcompile.bzl', 'tf_library')

tf_library(
    name = 'graph',
    config = 'graph.config.pbtxt',
    cpp_class = 'Graph',
    graph = 'graph.pb',
)

cc_binary(
    name = "libmodel.so",
    srcs = ["graph.cc"],
    deps = [":graph", "//third_party/eigen3"],
    linkopts = ["-lpthread"],
    linkshared = 1,
    copts = ["-fPIC"],
)

import numpy as np

print('Starting script')
libmodel = np.ctypeslib.load_library('libmodel', '/tensorflow/bazel-bin/external/org_tensorflow/')

libmodel.run.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(1,2), flags=('c', 'a')),
np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(1,1), flags=('c', 'a', 'w')),
np.ctypeslib.ctypes.c_float,
np.ctypeslib.ctypes.c_float]

x = np.require(np.zeros((1,2)), np.float32, ('c', 'a'))
y = np.require(np.zeros((1,1)), np.float32, ('c', 'a', 'w'))

res = libmodel.run(x, y, x.size, y.size)
print(res)

x = np.require(np.zeros((1,2)), np.float32, ('c', 'a'))
y = np.require(np.zeros((1,1)), np.float32, ('c', 'a', 'w'))
res = libmodel.run(x, y, x.size, y.size)