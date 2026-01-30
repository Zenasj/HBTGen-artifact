from tensorflow.keras import layers

from tensorflow import keras
from statistics import mean
import psutil
mem_stats=[]
for i in range(0, 10001):
  keras.backend.clear_session()
  a = keras.layers.Input(shape=(32,))
  b = keras.layers.Dense(32)(a)
  model = keras.Model(inputs=a, outputs=b)
  mem_stats.append(psutil.virtual_memory().used)
  if (i % 1000 == 0):
    print('memory -> step %5d Max %d Min %d Mean %d ' % (i, 
                                                         max(mem_stats)/(1024), 
                                                         min(mem_stats)/(1024), 
                                                         mean(mem_stats)/(1024)
                                                         )
    )
    mem_stats.clear()