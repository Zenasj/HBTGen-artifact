from segment_anything import build_sam, SamAutomaticMaskGenerator
import cv2
from torch import _dynamo as dynamo
image = cv2.imread('astronaut.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="../segment-anything/sam_vit_h_4b8939.pth"))

def func(image):
    return mask_generator.generate(image)

ex = dynamo.explain(func, image)[-1]
print(ex)