import torch
import numpy as np
import matplotlib.pyplot as plt

MODLE_LOCATION = "./models/mfi_0.97400.pth"
MODLE_LOCATION_QUAN = "./models/quantized_1_model.pth"
TENSOR_NAME = "fc1.weight"

def plot_distribution(model_name, tensor_set, resolution):
    model = torch.load(model_name)
    print(model)
    params = model.state_dict()
    tensor_value = params[TENSOR_NAME]
    tensor_value_np = tensor_value.numpy()
    tensor_value_np = tensor_value_np.flatten()
    bins = np.arange(-1, 1, resolution)
    plt.hist(tensor_value_np,bins) 
    plt.title("histogram") 
    plt.show()


if __name__ == '__main__':
    plot_distribution(MODLE_LOCATION, TENSOR_NAME, 0.01)
    plot_distribution(MODLE_LOCATION_QUAN, TENSOR_NAME, 0.01)