import numpy
import torch
import time
import torch.multiprocessing as mp
import queue
from PIL import Image

def myproducer(thequeue):
   for i in range(100):
      thearray=numpy.full((2048,2048),255,dtype=numpy.uint8)
      thequeue.put(torch.from_numpy(thearray))

def mywriter(thequeue):
   filecounter=0
   starttime=0
   while True:
      try:
         mytensor=thequeue.get_nowait()
         filecounter=filecounter+1
         if filecounter==1:
            starttime=time.time()
         mynumpyarray=mytensor.numpy()
         Image.fromarray(mynumpyarray).convert("L").save(str(filecounter)+".bmp","BMP")
         if filecounter==100:
            print(time.time()-starttime)
            break
      except queue.Empty:
         pass

# On ubuntu 
torch.multiprocessing.set_sharing_strategy("file_system")

thequeue=mp.Queue()
myproducerproc=mp.Process(target=myproducer,args=(thequeue,))
myproducerproc.start()
mywriterproc=mp.Process(target=mywriter,args=(thequeue,))
mywriterproc.start()
myproducerproc.join()
mywriterproc.join()