import numpy as np

#MVP String-mapping
import onnx

amap = {str(i): i for i in range(50)}
n = onnx.helper.make_node(
    'LabelEncoder',
    inputs=['X'],
    outputs=['Y'],
    keys_strings=amap.keys(),
    values_int64s=amap.values(), 
    domain='ai.onnx.ml'
)

X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.STRING, [None])
Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.INT64, [None])

graph_def = onnx.helper.make_graph(
    nodes=[n],
    name="mapper",
    inputs=[X],
    outputs=[Y]
)

model = onnx.helper.make_model(
    graph_def,
    opset_imports=[
        onnx.helper.make_opsetid('ai.onnx.ml', 2), 
        onnx.helper.make_opsetid('', 14)],
)

out_path = "data/modified_model.onnx"
onnx.checker.check_model(model)
onnx.save(model, out_path)

ort_session = onnxruntime.InferenceSession(out_path)
inp = np.array(["5", "11"])
ort_inputs = {ort_session.get_inputs()[0].name: inp}
ort_session.run(None, ort_inputs)