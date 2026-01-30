import sys 
import os
import tensorflow as tf
import gc
import time

class InceptionV3Graph:
    def __init__(self, graph_path):
        with tf.gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=tf.get_default_graph())

    def close(self):
        tf.reset_default_graph()
        gc.collect()
        self.sess.close()

if __name__ == "__main__":
    graph_path = "/path/to/classify_image_graph_def.pb" # you can get classify_image_graph_def.pb from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    N_GRAPHS = 100 
    graphs = dict()
    for i in range(N_GRAPHS):
        print("Loading graph {}".format(i+1))
        graphs[i] = InceptionV3Graph(graph_path)
        # If you uncomment these two lines below there won't be any the memory leak
        #graphs[i].close()
        #del graphs[i]
    for i in range(N_GRAPHS):
        print("Unloading graph {}".format(i+1))
        if i in graphs:
            graphs[i].close()
            del graphs[i]
    print(graphs)
    gc.collect()
    print("All graphs unloaded")
    time.sleep(120)
    print("Quitting...")