import torch.nn as nn
import torchvision

import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.ops import nms

def main():
    cap = cv2.VideoCapture(0)
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    t = transforms.ToTensor()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = t(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(x)[0]
        b, l, s = o["boxes"], o["labels"], o["scores"]
        i = (l == 1)
        b, s = b[i], s[i]
        k = nms(b, s, 0.7)
        b, s = b[k].cpu().numpy(), s[k].cpu().numpy()
        for box, score in zip(b, s):
            box = box.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            cv2.putText(frame, str(round(score.item(), 2)), (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

{python}
import torch

print("------- PyTorch Information -------")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Available CUDA Devices: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")

# Create a random tensor
x = torch.rand(5, 3)
print(f"Tensor Device: {x.device}")

# Move to GPU
x = x.cuda()
print(f"New Tensor Device: {x.device}")

# Enable benchmark mode
torch.backends.cudnn.benchmark = True

# Check memory usage
print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Reserved Memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Clear the cache
torch.cuda.empty_cache()