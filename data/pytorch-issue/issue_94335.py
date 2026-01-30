import torch.nn as nn

import cv2
import torch
import torch.nn.functional as F

SIZE_HW = (247, 111)
SIZE_WH = (111, 247)

t = torch.rand(1, 17, 64, 64)


pt_result = F.interpolate(t, size=SIZE_HW, mode="bicubic", align_corners=True)
cv_result = torch.from_numpy(cv2.resize(t.squeeze(0).permute(1, 2, 0).numpy(), SIZE_WH, interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)


print("max abs diff:", (pt_result - cv_result).abs().max())
print("max rel diff:", ((1 - pt_result) / cv_result).abs().max())
print("MSE:", ((pt_result - cv_result) ** 2).mean())

import cv2
import torch
import torch.nn.functional as F

SIZE_HW = (247, 111)
SIZE_WH = (111, 247)

t = torch.randint(0,255, (1, 17, 64, 64)).to(torch.uint8)

pt_result = torch.clamp(F.interpolate(t.float(), size=SIZE_HW, mode="bicubic", align_corners=False), 0, 255).to(torch.uint8)
cv_result = torch.from_numpy(cv2.resize(t.squeeze(0).permute(1, 2, 0).numpy(), SIZE_WH, interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)



print("max abs diff:", (pt_result.float() - cv_result.float()).abs().max())
print("MSE:", ((pt_result.float() - cv_result.float()) ** 2).mean())

t = t/255.0

pt_result = torch.clamp(F.interpolate(t.float(), size=SIZE_HW, mode="bicubic", align_corners=False), 0, 1)
cv_result = torch.from_numpy(cv2.resize(t.squeeze(0).permute(1, 2, 0).numpy(), SIZE_WH, interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).unsqueeze(0)

print("max abs diff:", (pt_result.float() - cv_result.float()).abs().max())
print("MSE:", ((pt_result.float() - cv_result.float()) ** 2).mean())