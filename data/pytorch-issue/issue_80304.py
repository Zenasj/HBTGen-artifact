build_dir = "./"

base_file_name = "my_model"

bo.export_finn_onnx(my_model, (1, 3, 1024), build_dir + "/{}.onnx".format(base_file_name))