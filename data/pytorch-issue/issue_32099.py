import onnx
import onnx2keras

model = onnx.load_model("model.onnx")

k_model = onnx2keras.onnx_to_keras(model, ["input_data"])