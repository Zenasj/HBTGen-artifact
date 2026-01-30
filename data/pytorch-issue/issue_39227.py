import onnx
import onnxruntime as rt
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def compute_divide(self, x, img, index_x, index_y, proj_x, proj_y):

        device = torch.device("cuda")

        desired_channel = img.size(0)
        len_indices = index_x.size(0)

        x = x.squeeze(0)

        fast_idx = torch.arange(desired_channel).unsqueeze(1).expand(desired_channel, len_indices).transpose(1,
                                                                    0).flatten().to(device)

        y_max = img.size(1)
        x_max = img.size(2)

        unet_rv_img_orig_flat = img.flatten()

        proj_x_ex = proj_x.unsqueeze(1).expand(len_indices, desired_channel).flatten().to(device)
        proj_y_ex = proj_y.unsqueeze(1).expand(len_indices, desired_channel).flatten().to(device)

        new_indicies = (fast_idx * x_max * y_max) + (proj_y_ex * x_max) + proj_x_ex
        data = unet_rv_img_orig_flat.index_select(0, new_indicies.to(device))

        indx = (fast_idx,
                index_x.expand(len_indices, desired_channel).flatten().to(device),
                index_y.expand(len_indices, desired_channel).flatten().to(device))

        y = x.index_put(indx, data, accumulate=True)

        print("y_in_model",y)
        
        return y

    def forward(self, x, img, index_x, index_y, proj_x, proj_y):
        device = torch.device("cuda")

        z = self.compute_divide(x, img, index_x, index_y, proj_x, proj_y)

        print("out",z)

        return z

def main():
    model = Model()
    onnx_path = "test_slice.onnx"

    dummy_input = torch.zeros(1, 3, 5, 5)
    dummy_input = dummy_input.cuda()

    img = torch.randint(0,10,(3,6,7)).float().cuda()

    index_x = torch.tensor([[0], [4], [4], [4] ]).reshape(4,1).cuda() #
    index_y = torch.tensor([[3], [4], [3], [4] ]).reshape(4,1).cuda() #
    proj_x = torch.tensor([0, 6, 3, 6 ]).reshape(4,).cuda() #
    proj_y = torch.tensor([1, 5, 3, 5 ]).reshape(4,).cuda() #

    input_names=["x", "img", "index_x","index_y", "proj_x", "proj_y"]
    output_names=["out_"]

    inputs = (dummy_input, img, index_x, index_y, proj_x, proj_y)

    with torch.no_grad():

        torch.onnx.export(model,
                          inputs,
                          onnx_path,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=12,
                          )

    print("Done Exporting ONNX file")

    # Load the ONNX model
    model_test = onnx.load(onnx_path)

    # Check that the IR is well formed
    # Check the model
    try:
        onnx.checker.check_model(model_test)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    sess = rt.InferenceSession(onnx_path)

    inputs = sess.get_inputs()
    in_1 = inputs[0].name
    in_2 = inputs[1].name
    in_3 = inputs[2].name
    in_4 = inputs[3].name
    in_5 = inputs[4].name
    in_6 = inputs[5].name

    print("inputs[0]:",inputs[0].name,inputs[0].shape,inputs[0].type)
    print("inputs[1]:",inputs[1].name,inputs[1].shape,inputs[1].type)
    print("inputs[2]:",inputs[2].name,inputs[2].shape,inputs[2].type)

    # outputs = sess.get_outputs()
    # out_ = outputs[0].name

    pred = sess.run(None, {in_1: dummy_input.cpu().detach().numpy(),
                           in_2: img.cpu().detach().numpy(),
                           in_3: index_x.cpu().detach().numpy(),
                           in_4: index_y.cpu().detach().numpy(),
                           in_5: proj_x.cpu().detach().numpy(),
                           in_6: proj_y.cpu().detach().numpy(),
                           })

    print("pred",pred[0])
    print("FINALLY SUCCESS !!!!!")