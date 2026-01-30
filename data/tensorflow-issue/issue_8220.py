run_session('0')

run_session('0')
run_session('1')

import multiprocessing as mp

p = mp.Pool(2)
p.map(run_session, ['0', '1'])
p.close()
p.join()

import multiprocessing as mp

run_session('0')
p = mp.Pool(2)
p.map(run_session, ['0', '1'])
p.close()
p.join()

import os
import tensorflow
from multiprocessing.pool import Pool

def runInSubprocess(somearg):
    print('Training model on process id {}.'.format(os.getpid()))
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())

# This Hangs:
runInSubprocess(2)
Pool(processes=2).map(runInSubprocess, [1,2])

# This works:
runInSubprocess(2)
runInSubprocess(2)

# This works:
Pool(processes=2).map(runInSubprocess, [1,2])
Pool(processes=2).map(runInSubprocess, [1,2])

# This works:
Pool(processes=2).map(runInSubprocess, [1,2])
runInSubprocess(2)

import os
import multiprocessing


class Predictor(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, gpu_id):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gpu_id = gpu_id

    def run(self):
        #set GPU id before importing tensorflow!!!!!!!!!!!!!
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_id)
        #import tensorflow here
        import tensorflow as tf
        sess = tf.Session()
        print('Using device #%s' % self.gpu_id)
        a = tf.placeholder(tf.int16, name='a')
        y = tf.identity(a, name='y')
        while True:
            input = self.input_queue.get()
            if input is None:
                self.input_queue.task_done()
                print("Exiting Process %d" % self.gpu_id)
                break
            else:
                res = sess.run(y, feed_dict={a: input})
                self.input_queue.task_done()
                self.output_queue.put(res)
        sess.close()
        return

if __name__ == "__main__":
    jobs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_gpus = 2
    p_list = []
    input_queue = multiprocessing.JoinableQueue()
    output_queue = multiprocessing.Queue()
    for i in range(num_gpus):
        p = Predictor(input_queue, output_queue, i)
        p_list.append(p)

    for p in p_list:
        p.start()

    for job in jobs:
        input_queue.put(job)

    for i in range(num_gpus):
        input_queue.put(None)

    for i in range(num_gpus):
        print(output_queue.get())

    input_queue.join()
    
    for p in p_list:
        p.join()