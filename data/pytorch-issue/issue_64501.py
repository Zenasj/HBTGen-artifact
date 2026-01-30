import numpy as np
from PIL import Image
import cv2
from skimage.transform import resize

size = 10
a_in = np.arange(0, 3 * 32 * 32, dtype="uint8").reshape(32, 32, 3)

pil_in = Image.fromarray(a_in)
pil_out = pil_in.resize((size, size), resample=Image.NEAREST)
a_pil_out = np.array(pil_out)

a_cv2_out = cv2.resize(a_in, dsize=(size, size), interpolation=cv2.INTER_NEAREST_EXACT)

a_skimg_out = resize(a_in, (size, size), order=0, preserve_range=True, anti_aliasing=False).astype("uint8")

print(np.allclose(a_cv2_out, a_pil_out), np.mean(np.abs(a_cv2_out - a_pil_out)))
# > (False, 38.74)

print(np.allclose(a_cv2_out, a_skimg_out), np.mean(np.abs(a_cv2_out - a_skimg_out)))
# > (False, 25.3)