import random

import tensorflow as tf
import os
import numpy as np
try:
  pool_size_0 = 1e+38
  pool_size = [pool_size_0,]
  strides_0 = 2
  strides = [strides_0,]
  padding = "same"
  data_format = "channels_last"
  arg_class = tf.compat.v1.layers.MaxPooling1D(pool_size=pool_size,strides=strides,padding=padding,data_format=data_format,)
  arg_input_0_tensor = tf.random.uniform([1, 5, 4], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

### Relevant log output



{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.866 NotebookApp] Searching ['/root/.jupyter', '/root/.local/etc/jupyter', '/usr/etc/jupyter', '/usr/local/etc/jupyter', '/etc/jupyter'] for config files","time":"2023-08-21T05:53:09.867Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.866 NotebookApp] Searching ['/root/.jupyter', '/root/.local/etc/jupyter', '/usr/etc/jupyter', '/usr/local/etc/jupyter', '/etc/jupyter'] for config files","time":"2023-08-21T05:53:09.870Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.866 NotebookApp] Looking for jupyter_config in /etc/jupyter","time":"2023-08-21T05:53:09.868Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.878 NotebookApp] Looking for jupyter_config in /usr/local/etc/jupyter","time":"2023-08-21T05:53:09.882Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.883 NotebookApp] Looking for jupyter_config in /usr/etc/jupyter","time":"2023-08-21T05:53:09.887Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.883 NotebookApp] Looking for jupyter_config in /root/.local/etc/jupyter","time":"2023-08-21T05:53:09.887Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.884 NotebookApp] Looking for jupyter_config in /root/.jupyter","time":"2023-08-21T05:53:09.887Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.866 NotebookApp] Looking for jupyter_config in /etc/jupyter","time":"2023-08-21T05:53:09.871Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.869 NotebookApp] Looking for jupyter_config in /usr/local/etc/jupyter","time":"2023-08-21T05:53:09.873Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.870 NotebookApp] Looking for jupyter_config in /usr/etc/jupyter","time":"2023-08-21T05:53:09.882Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.870 NotebookApp] Looking for jupyter_config in /root/.local/etc/jupyter","time":"2023-08-21T05:53:09.882Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.871 NotebookApp] Looking for jupyter_config in /root/.jupyter","time":"2023-08-21T05:53:09.883Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.871 NotebookApp] Looking for jupyter_notebook_config in /etc/jupyter","time":"2023-08-21T05:53:09.883Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.875 NotebookApp] Loaded config file: /etc/jupyter/jupyter_notebook_config.py","time":"2023-08-21T05:53:09.884Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.876 NotebookApp] Looking for jupyter_notebook_config in /usr/local/etc/jupyter","time":"2023-08-21T05:53:09.886Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.876 NotebookApp] Loaded config file: /usr/local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:09.886Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.877 NotebookApp] Looking for jupyter_notebook_config in /usr/etc/jupyter","time":"2023-08-21T05:53:09.887Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.877 NotebookApp] Looking for jupyter_notebook_config in /root/.local/etc/jupyter","time":"2023-08-21T05:53:09.888Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.877 NotebookApp] Looking for jupyter_notebook_config in /root/.jupyter","time":"2023-08-21T05:53:09.891Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.879 NotebookApp] Loaded config file: /root/.jupyter/jupyter_notebook_config.py","time":"2023-08-21T05:53:09.891Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.889 NotebookApp] Looking for jupyter_notebook_config in /etc/jupyter","time":"2023-08-21T05:53:09.889Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.889 NotebookApp] Loaded config file: /etc/jupyter/jupyter_notebook_config.py","time":"2023-08-21T05:53:09.891Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.890 NotebookApp] Looking for jupyter_notebook_config in /usr/local/etc/jupyter","time":"2023-08-21T05:53:09.892Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.890 NotebookApp] Loaded config file: /usr/local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:09.892Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.890 NotebookApp] Looking for jupyter_notebook_config in /usr/etc/jupyter","time":"2023-08-21T05:53:09.892Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.890 NotebookApp] Looking for jupyter_notebook_config in /root/.local/etc/jupyter","time":"2023-08-21T05:53:09.892Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.890 NotebookApp] Looking for jupyter_notebook_config in /root/.jupyter","time":"2023-08-21T05:53:09.892Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"[D 05:53:09.891 NotebookApp] Loaded config file: /root/.jupyter/jupyter_notebook_config.py","time":"2023-08-21T05:53:09.895Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.735Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/local/etc/jupyter/jupyter_notebook_config.d/panel-client-jupyter.json","time":"2023-08-21T05:53:10.745Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.745Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.746Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/root/.local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.747Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/root/.jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.747Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret","time":"2023-08-21T05:53:10.765Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Authentication of /metrics is OFF, since other authentication is disabled.","time":"2023-08-21T05:53:10.771Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"google.colab serverextension initialized.","time":"2023-08-21T05:53:10.836Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.878Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/local/etc/jupyter/jupyter_notebook_config.d/panel-client-jupyter.json","time":"2023-08-21T05:53:10.880Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.880Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/usr/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.882Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/root/.local/etc/jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.883Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"    \t/root/.jupyter/jupyter_notebook_config.json","time":"2023-08-21T05:53:10.884Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret","time":"2023-08-21T05:53:10.893Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Authentication of /metrics is OFF, since other authentication is disabled.","time":"2023-08-21T05:53:10.894Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"google.colab serverextension initialized.","time":"2023-08-21T05:53:10.945Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Serving notebooks from local directory: /","time":"2023-08-21T05:53:15.513Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Jupyter Notebook 6.5.5 is running at:","time":"2023-08-21T05:53:15.514Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Serving notebooks from local directory: /","time":"2023-08-21T05:53:15.521Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Jupyter Notebook 6.5.5 is running at:","time":"2023-08-21T05:53:15.521Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"http://172.28.0.2:9000/","time":"2023-08-21T05:53:15.521Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).","time":"2023-08-21T05:53:15.522Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"http://172.28.0.12:9000/","time":"2023-08-21T05:53:15.523Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).","time":"2023-08-21T05:53:15.523Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"Kernel started: 7be3ebb5-8079-41dd-a058-72f1e271f55f, name: python3","time":"2023-08-21T05:53:32.402Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:38.426472: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.","time":"2023-08-21T05:53:38.426Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.","time":"2023-08-21T05:53:38.426Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:39.614903: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT","time":"2023-08-21T05:53:39.615Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:43.858531: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:43.858Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:44.410103: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:44.410Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:44.410567: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:44.410Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:44.413324: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:44.413Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:44.413957: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:44.414Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:44.414476: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:44.414Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:48.373843: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:48.374Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:48.380854: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:48.381Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:48.381791: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355","time":"2023-08-21T05:53:48.381Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:48.382383: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.","time":"2023-08-21T05:53:48.382Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:48.382464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13664 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5","time":"2023-08-21T05:53:48.382Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:53.331521: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8900","time":"2023-08-21T05:53:53.331Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"2023-08-21 05:53:53.332580: F tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:959] Check failed: cudnnSetPoolingNdDescriptor( handle_.get(), (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum ? cudnn_max_pooling_mode : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING), propagate_nans ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN, nd, shape.data(), padding.data(), strides.data()) == CUDNN_STATUS_SUCCESS (3 vs. 0)","time":"2023-08-21T05:53:53.332Z","v":0}
{"pid":7,"type":"jupyter","level":30,"msg":"KernelRestarter: restarting kernel (1/5), keep random ports","time":"2023-08-21T05:53:56.402Z","v":0}
{"pid":7,"type":"jupyter","level":40,"msg":"WARNING:root:kernel 7be3ebb5-8079-41dd-a058-72f1e271f55f restarted","time":"2023-08-21T05:53:56.402Z","v":0}