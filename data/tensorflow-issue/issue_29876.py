import multiprocessing

class AlignProcess (multiprocessing.Process):
    def __init__(self, gpu ):
        multiprocessing.Process.__init__(self)
        self.gpu = gpu
      
def run(self):
    if self.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(self.gpu - 1)

        import tensorflow as tf 
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        session.close()


# Later on in code

AlignProcess(1).start()