source_path_pre = '/home/memo/PycharmProjects/ONNX/resnet50_pre.pb'
source_path_init = '/home/memo/PycharmProjects/ONNX/resnet50_init.pb'
destination_path = '/home/memo/PycharmProjects/ONNX/resnet50.onnx'

data_type = onnx.TensorProto.FLOAT
data_shape = (1, 3, 224, 224)
value_info = {
    'data': (data_type, data_shape)
}

predict_net = caffe2_pb2.NetDef()
with open(source_path_pre, "rb") as predict_stream:
    predict_net.ParseFromString(predict_stream.read())
if predict_net.name == '':
	predict_net.name = 'modelName'

init_net = caffe2_pb2.NetDef()
with open(source_path_init, "rb") as init_stream:
	init_net.ParseFromString(init_stream.read())


onnx_model = c2_onnx.caffe2_net_to_onnx_model(predict_net=predict_net,
                                              init_net=init_net,
                                              value_info=value_info)
with open(destination_path, 'wb') as f:
	f.write(onnx_model.SerializeToString())