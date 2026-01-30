import torch
import torch.nn as nn

warped_right_feature_map_a = torch.gather(right_input,dim=3, index=right_y_coordinate_a.long())

import onnx
onnx_model = onnx.load("IresNet-1.onnx")
onnx.checker.check_model(onnx_model)

torch.onnx.export(learn.model,               # model being run
                  (left,right),                         # model input (or a tuple for multiple inputs)
                  "Test.onnx",   # where to save the model (can be a file or file-like object)
                  verbose = True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  input_names = ['input1','input2'],   # the model's input names
                  output_names = ['output1'], # the model's output names
                    )

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, right_input, disparity_samples):
        """
        Args:
            :right_input: Right Image Features,with shape [B,C,H,W],in the Inference mode:the real shape is [1,32,704,1280]
            :disparity_samples:  Disparity,with shape [B,C,H,W],in the Inference mode:the real shape is [1,1,704,1280]
        Returns:
            :warped_right_feature_map: right image features warped according to input disparity.,
            with shape [B,C,H,W],in the Inference mode:the real shape is [1,32,704,1280]
        """
        device = right_input.get_device()
        # B, C, H, W = right_input.shape
		#[H,W]
        left_y_coordinate = torch.arange(0.0, 1280).repeat(704)
        #[H,W]
        left_y_coordinate = left_y_coordinate.view(704, 1280)

        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max= 1280 - 1)
        #[B,H,W]
        # left_y_coordinate = left_y_coordinate.expand(1, -1, -1)
        # left_y_coordinate = left_y_coordinate.expand(1,  -1,  -1, -1)
        #[B,H,W]
        left_y_coordinate = left_y_coordinate.unsqueeze(0)
        #[B,C,H,W]
        left_y_coordinate = left_y_coordinate.unsqueeze(0)
        

        right_y_coordinate = left_y_coordinate + disparity_samples

	
		#[B,C,H,W]
        right_y_coordinate_a = torch.floor(right_y_coordinate)
        right_y_coordinate_b = right_y_coordinate_a + 1
		
        wa = right_y_coordinate_b - right_y_coordinate
        wb = right_y_coordinate   - right_y_coordinate_a

        wa = wa.repeat(1, 32, 1, 1)
        wb = wb.repeat(1, 32, 1, 1)

        right_y_coordinate_a = right_y_coordinate_a.repeat(1, 32, 1, 1).long()
        right_y_coordinate_b = right_y_coordinate_b.repeat(1, 32, 1, 1).long()
        #[B,C,H,W]
        right_y_coordinate_a = torch.clamp(right_y_coordinate_a.float(), min=0, max= 1280 - 1)
        right_y_coordinate_b = torch.clamp(right_y_coordinate_b.float(), min=0, max= 1280 - 1)
        #test_index = torch.zeros_like(right_input)


        warped_right_feature_map_a = torch.gather(right_input, dim=3, index=right_y_coordinate_a.long())
        warped_right_feature_map_b = torch.gather(right_input, dim=3, index=right_y_coordinate_b.long())	
		
        warped_right_feature_map = wa * warped_right_feature_map_a + wb * warped_right_feature_map_b

        # #right_y_coordinate_1 = right_y_coordinate_1.expand(right_input.size()[1], -1, -1, -1).permute([1, 0, 2, 3]))
        right_y_coordinate_1 = right_y_coordinate.repeat(1, 32, 1, 1)
        warped_right_feature_map = (1.0 - ((right_y_coordinate_1 < 0).float() +
                                   (right_y_coordinate_1 > torch.tensor([1280 - 1], dtype=torch.float32))).float()) * \
            (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)
        return warped_right_feature_map

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, right_input, disparity_samples):
        B, C, H, W = right_input.shape
        device = right_input.get_device()
        left_y_coordinate = torch.arange(0.0, W).repeat(H)

        left_y_coordinate = left_y_coordinate.view(H, W)

        left_y_coordinate = left_y_coordinate.unsqueeze(0)
        left_y_coordinate = left_y_coordinate.unsqueeze(0)
        

        right_y_coordinate = left_y_coordinate + disparity_samples

        right_y_coordinate_a = torch.floor(right_y_coordinate)
        right_y_coordinate_b = right_y_coordinate_a + 1
		
        wa = right_y_coordinate_b - right_y_coordinate
        wb = right_y_coordinate   - right_y_coordinate_a

        wa = wa.repeat(1, C, 1, 1)
        wb = wb.repeat(1, C, 1, 1)

        right_y_coordinate_a = right_y_coordinate_a.repeat(1, C, 1, 1).long()
        right_y_coordinate_b = right_y_coordinate_b.repeat(1, C, 1, 1).long()


        right_y_coordinate_a = torch.clamp(right_y_coordinate_a, min=0, max= W - 1)
        right_y_coordinate_b = torch.clamp(right_y_coordinate_b, min=0, max= W - 1)

        warped_right_feature_map_a = torch.ones_like(right_input)
        warped_right_feature_map_b = torch.ones_like(right_input)


        warped_right_feature_map_a = right_input.gather(dim=3, index=right_y_coordinate_a.long())
        warped_right_feature_map_b = torch.gather(right_input, dim=3, index=right_y_coordinate_b.long())	
		
        warped_right_feature_map = wa * warped_right_feature_map_a + wb * warped_right_feature_map_b

        right_y_coordinate_1 = right_y_coordinate.repeat(1, C, 1, 1)
        
        warped_right_feature_map = (1.0 - ((right_y_coordinate_1 < 0).float() +
                                   (right_y_coordinate_1 > torch.tensor([W - 1], dtype=torch.float32))).float()) * \
            (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        print("new stn")
        return warped_right_feature_map