with tf.compat.v1.enable_control_flow_v2():
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('/home/marat/OCR/yolo3/model120.pb', ['test_input'], ['test_output'])
        tflite_model = converter.convert()
        open("model120.tflite", "wb").write(tflite_model)

import torch
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

img_tensor = torch.zeros(1, 3, 128, 128)

if 1:
    model = SimpleNet().eval()
    dummy_output = model(img_tensor)
    print(dummy_output.detach().cpu().numpy().shape, dummy_output.detach().cpu().numpy())
    torch.onnx.export(model, img_tensor, './dummy.onnx', input_names=['test_input'], output_names=['test_output'])

if 1:
    import onnx
    import tensorflow.compat.v1 as tf
    tf.compat.v1.enable_control_flow_v2()
    from onnx_tf.backend import prepare
    model_onnx = onnx.load('./dummy.onnx')
    tf_rep = prepare(model_onnx)
    tf_rep.export_graph('./dummy.pb')

if 1:
    import tensorflow as tf

    def load_pb(path_to_pb):
        with tf.compat.v1.gfile.GFile(path_to_pb, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.compat.v1.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name='')
        return graph, graph_def

    tf_graph, graph_def = load_pb('./dummy.pb')
    sess = tf.compat.v1.Session(graph=tf_graph)

    # Show tensor names in graph
    for op in tf_graph.get_operations():
        print(op.values())

    output_tensor = tf_graph.get_tensor_by_name('test_output:0')
    input_tensor = tf_graph.get_tensor_by_name('test_input:0')

    dummy_input = img_tensor.numpy()

    output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})

    print('TF output')
    print(output.shape)
    print(output)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('/home/marat/OCR/yolo3/dummy.pb', ['test_input'], ['test_output'])
    tflite_model = converter.convert()
    open("model120.tflite", "wb").write(tflite_model)

for n in sess.graph.get_operations():
     print(n)