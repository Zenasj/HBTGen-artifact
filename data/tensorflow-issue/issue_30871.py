import numpy as np
import random
import tensorflow as tf

def run_calibration(calib_graph, dataset):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    x = np.random.randint(0,10000,(1,40))
    with tf.Graph().as_default() as g:
        input, output = tf.import_graph_def(graph_def=calib_graph, return_elements=["X", "model/dense/BiasAdd"],
                                            name='')
        input = input.outputs[0]
        output = output.outputs[0]
        sess = tf.Session(config=tf_config, graph=g)

        for i in range(10):
            val = sess.run(output, {input: x})
        return calib_graph


def int8quant():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600.meta")
        saver.restore(sess, "/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600")
        your_outputs = ['model/dense/BiasAdd']
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=your_outputs)
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=your_outputs,
            max_batch_size=10,
            max_workspace_size_bytes=2 << 30,
            precision_mode='INT8',
            minimum_segment_size=2  # minimum number of nodes in an engine
       )
    int8graph = run_calibration(trt_graph, None)
    int8_graph = trt.calib_graph_to_infer_graph(int8graph)

with tf.Session() as sess:
        saver = tf.train.import_meta_graph("/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600.meta")
        saver.restore(sess, "/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600")
        outputs = ['model/dense/BiasAdd']
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=outputs)
        converter = trt_convert.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=outputs,
            precision_mode=trt_convert.TrtPrecisionMode.INT8,
        )

        class CalibrationData(object):

            def __init__(self,dataset=None):
                self.dataset = dataset

            def next(self):
                X =  np.random.randint(0, 10000, (1, 40))
                return {"X:0": X}

        calib_graph = converter.convert()
        calib_graph = converter.calibrate(
            fetch_names=["model/dense/BiasAdd:0"],
            num_runs=10,
            feed_dict_fn=CalibrationData().next)
        tf.train.write_graph(calib_graph, "/workspace/exps/ptb/ptb_rnn_128", "trt_model_int8.pb", as_text=False)