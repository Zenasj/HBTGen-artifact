import torch

@torch.jit.script
def image_pool(img, crop_size,
               input_x, input_y, input_width, input_height,
               target_width, target_height):#, target_top, target_left):
    #outputs = []
    batch = 0 #input_x.shape[0] - 1
    idx = -1
    while idx < batch:
        idx = idx + 1
        y = input_y[idx]
        x = input_x[idx]
        h = input_height[idx]
        w = input_width[idx]
        new_h = target_height[idx]
        new_w = target_width[idx]
        crop = img[:, :, 0:y+h+crop_size, 0:x+w+crop_size]
        resized_crop = crop[:, :, :crop_size, :crop_size]
        return resized_crop

@torch.jit.script
def image_pool(img, crop_size,
               input_x, input_y, input_width, input_height,
               target_width, target_height):#, target_top, target_left):
    #outputs = []
    batch = 0 #input_x.shape[0] - 1
    idx = -1
    idx = idx + 1
    y = input_y[idx]
    x = input_x[idx]
    h = input_height[idx]
    w = input_width[idx]
    new_h = target_height[idx]
    new_w = target_width[idx]
    crop = img[:, :, 0:y+h+crop_size, 0:x+w+crop_size]
    resized_crop = crop[:, :, :crop_size, :crop_size]
    return resized_crop