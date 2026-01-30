import numpy as np
import tensorflow as tf

tf.io.decode_image

def get_item(type, index):
    if type == 'trainval':
        item = images_trainval[index]
    else:
        item = images_val[index]

    image = item['image/encoded']
    image = tf.io.decode_image(image, 3)
    image = image.numpy()

    mask = item['image/segmentation/class/encoded']
    mask = tf.io.decode_image(mask, 1)
    mask = mask.numpy()
    mask = mask.reshape(mask.shape[:2])

    return image, mask

class DLDataset(Dataset):
    def __init__(self, split, dataset_dir):
        self.split = split
        
        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        image, mask = get_item(self.split, i)

        image, mask = safe_crop(image, mask, size=512)
        image = transforms.ToPILImage()(image.copy().astype(np.uint8))
        image = self.transformer(image)

        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return get_len(self.split)

tf.io.decode_jpeg

tf.io.decode_png

tf.io.decode_image

def get_item(type, index):
    if type == 'trainval':
        item = images_trainval[index]
    else:
        item = images_val[index]

    image = item['image/encoded']
    image = tf.io.decode_jpeg(image, 3)
    image = image.numpy()

    mask = item['image/segmentation/class/encoded']
    mask = tf.io.decode_png(mask, 1)
    mask = mask.numpy()
    mask = mask.reshape(mask.shape[:2])

    return image, mask

workers > 1

use_multiprocessing = True

tf.io.decode_image

torch.utils.data.Dataset