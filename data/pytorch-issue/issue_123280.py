import torch
import zipfile
import os


def load_pt2_model_from_local_zip(zip_file_path: str, device: str):
    """Function to unzip a local ZIP file and load a model.

    Args:
        zip_file_path (str): Local path of the ZIP file containing the model.
        device (str): The device to load the model onto.

    Returns:
        Callable: The loaded model.
    """
    zip_file_name = os.path.basename(zip_file_path)
    extract_dir_base = zip_file_name.rsplit(".", 1)[0]
    save_dir = os.path.join("/opt/workspace", extract_dir_base)

    os.makedirs(save_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    model = torch._export.aot_load("/opt/workspace/aot_inductor_gpu_tensor_cores/solar_satlas_sentinel2_model_pt2.so", "cuda") # segfault error is here

    return model


model1 = torch._export.aot_load("/opt/workspace/aot_inductor_gpu_tensor_cores/solar_satlas_sentinel2_model_pt2.so", "cuda")
#unzips the .so and .cubin files over the previous files
model2 = load_pt2_model_from_local_zip("/opt/workspace/aot_inductor_gpu_tensor_cores.zip", "cuda") # segfault error is here

print("done") # never reached