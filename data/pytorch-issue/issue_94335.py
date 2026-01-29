# torch.rand(1, 17, 64, 64, dtype=torch.float32)
import torch
import cv2
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def forward(self, x):
        # PyTorch bicubic interpolation
        pt_x = x.float()
        pt_result = F.interpolate(pt_x, size=(247, 111), mode='bicubic', align_corners=False)
        
        # OpenCV bicubic interpolation
        b, c, h, w = x.shape
        cv_results = []
        for i in range(b):
            # Convert to numpy and process OpenCV
            img = x[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            resized = cv2.resize(img, (111, 247), interpolation=cv2.INTER_CUBIC)  # (247, 111, C)
            resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).float()  # (C, 247, 111)
            cv_results.append(resized_tensor.to(x.device))
        cv_result = torch.stack(cv_results)
        
        # Compute maximum absolute difference
        diff = pt_result - cv_result
        max_abs_diff = diff.abs().max()
        return max_abs_diff

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 17, 64, 64, dtype=torch.float32)

