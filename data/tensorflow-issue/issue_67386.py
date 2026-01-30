import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras

class Seg3DRandomDataset(tf.keras.utils.Sequence):
    def __init__(self, images_dir, labels_dir, input_size, num_classes, batch_size,
                 hu_min_val, hu_max_val, mode:str,**kwargs):
        super(Seg3DRandomDataset, self).__init__(**kwargs)

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hu_min_val = hu_min_val
        self.hu_max_val = hu_max_val
        self.mode = mode
    
        file_names = os.listdir(images_dir) # nii.gz
        

        self.images_path = []
        self.labels_path = []
        for file_name in file_names:
            image_path = os.path.join(images_dir, file_name)
            label_path = os.path.join(labels_dir, file_name)
            self.images_path.append(image_path)
            self.labels_path.append(label_path)
        
    def __len__(self):
        return math.ceil(len(self.images_path) / self.batch_size)
    
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.images_path))

        batch_x_path = self.images_path[low:high]
        batch_y_path = self.labels_path[low:high]
        
        batch_x = []
        batch_y = []
        for x_path, y_path in zip(batch_x_path, batch_y_path):
            image = sitk.ReadImage(x_path)
            label = sitk.ReadImage(y_path)
            self.curimagePath = x_path

            
            image = self.custom_window(image,self.hu_min_val,self.hu_max_val)
            # 
            imageArray = sitk.GetArrayFromImage(image)
            labelArray = sitk.GetArrayFromImage(label)

            
            self.d_range, self.h_range, self.w_range = self.cal_board(labelArray)

            
            imageArray = self.normalize_imageArray(imageArray[self.d_range[0]:self.d_range[1]+1, self.h_range[0]:self.h_range[1]+1, self.w_range[0]:self.w_range[1]+1])
            labelArray = labelArray[self.d_range[0]:self.d_range[1]+1, self.h_range[0]:self.h_range[1]+1, self.w_range[0]:self.w_range[1]+1]

            
            imageArray, labelArray = self.random_crop(imageArray, labelArray, self.input_size)

            if self.mode == 'train':
                if random.randint(0, 1) == 1:                    
                    imageArray = np.flip(imageArray, axis=1)
                    labelArray = np.flip(labelArray, axis=1)
                if random.randint(0, 1) == 1:
                    imageArray = np.flip(imageArray, axis=2)
                    labelArray = np.flip(labelArray, axis=2)
                if random.randint(0, 1) == 1:
                    rotate_angle = random.randint(0, 360)
                    imageArray = ndimage.rotate(imageArray, rotate_angle, axes=[1,2],reshape=False,mode='nearest',order=0)
                    labelArray = ndimage.rotate(labelArray, rotate_angle, axes=[1,2],reshape=False,mode='nearest',order=0)
            

           
            # ->[1,d,h,w]
            img_c = np.expand_dims(imageArray, axis=-1).astype(np.float32)
            
            # labelArray onehot ->[d,h,w,num_classes]
            lab_c = np.zeros((*labelArray.shape,self.num_classes),dtype=np.float32)
            for i in range(self.num_classes):
                lab_c[...,i] = (labelArray == i).astype(np.float32)
           
            batch_x.append(img_c)
            batch_y.append(lab_c)

        
        return np.array(batch_x), np.array(batch_y)

    def random_crop(self, imageArray, labelArray, crop_size):
        D,H,W = imageArray.shape
        d,h,w = crop_size

        if D<d:
            padding = np.zeros((d-D,H,W),dtype=np.float32)
            imageArray = np.concatenate((imageArray, padding), axis=0)
            labelArray = np.concatenate((labelArray, padding), axis=0)
        if H<h:
            D,H,W = imageArray.shape
            padding = np.zeros((D,h-H,W),dtype=np.float32)
            imageArray = np.concatenate((imageArray, padding), axis=1)
            labelArray = np.concatenate((labelArray, padding), axis=1)
        if W<w:
            D,H,W = imageArray.shape
            padding = np.zeros((D,H,w-W),dtype=np.float32)
            imageArray = np.concatenate((imageArray, padding), axis=2)
            labelArray = np.concatenate((labelArray, padding), axis=2)
        
        D,H,W = imageArray.shape

        d_start = random.randint(0, D-1)
        d_end = d_start + d
        if d_end > D:
            d_end = D
            d_start = D - d

        h_start = random.randint(0, H-1)
        h_end = h_start + h
        if h_end > H:
            h_end = H
            h_start = H - h

        w_start = random.randint(0, W-1)
        w_end = w_start + w
        if w_end > W:
            w_end = W
            w_start = W - w
        
        img = imageArray[d_start:d_end, h_start:h_end, w_start:w_end]
        lab = labelArray[d_start:d_end, h_start:h_end, w_start:w_end]
        assert img.shape == (d,h,w) and lab.shape==(d,h,w), f"img.shape={img.shape}, lab.shape={lab.shape}, (d,h,w)={crop_size}"
            
        return img, lab
    

    def normalize_imageArray(self, image_array):   
        max_value = np.max(image_array)
        min_value = np.min(image_array)
        assert max_value==self.hu_max_val and min_value==self.hu_min_val,f"max_value={max_value}, hu_max_val={self.hu_max_val}; min_value={min_value}, hu_min_val={self.hu_min_val}, they are not equal!, image_path={self.curimagePath}"

        img_01 = (image_array - min_value) / (max_value - min_value)
        return np.clip(img_01, 0, 1)
    

    def cal_board(self, label_array):
        dots = np.argwhere(label_array != 0)
        mins = np.min(dots, axis=0)
        maxs = np.max(dots, axis=0)

        d_range = [mins[0], maxs[0]]
        h_range = [mins[1], maxs[1]]
        w_range = [mins[2], maxs[2]]
        
        return d_range, h_range, w_range


    def custom_window(self,image,hu_min_val,hu_max_val):
        ww_filter = sitk.IntensityWindowingImageFilter()
        ww_filter.SetWindowMinimum(hu_min_val)
        ww_filter.SetWindowMaximum(hu_max_val)

        ww_filter.SetOutputMinimum(hu_min_val)
        ww_filter.SetOutputMaximum(hu_max_val)
        return ww_filter.Execute(image)

    def on_epoch_end(self):
        if self.mode == 'train':
            seed = random.randint(1,100)
            random.seed(seed)
            random.shuffle(self.images_path)
            random.seed(seed)
            random.shuffle(self.labels_path)

val_images_dir = "D:/data/lung/images/val"
val_labels_dir = "D:/data/lung/labels/val"
val_dataset = Seg3DRandomDataset(
    images_dir=val_images_dir,labels_dir=val_labels_dir,
    input_size=input_size,num_classes=num_classes,  
    batch_size=batch_size,   
    hu_min_val=-1000,hu_max_val=800,
    mode="val"
)
for x,y in val_dataset:
        print(x.shape,y.shape)
        if(x.shape[0]==0):
            input("zzzz")

input("zzz2")