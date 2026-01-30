import numpy as np
import tensorflow as tf
from tensorflow import keras

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=BATCH_SIZE, shuffle=True):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = data.index.tolist()

    # @staticmethod
    def __load_dicom_image(self,path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        if rotate > 0:
            rot_choices = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            data = cv2.rotate(data, rot_choices[rotate])

        data = cv2.resize(data, (img_size, img_size))
        return data

    

    def __load_dicom_images_3d(self, scan_id, num_imgs=NUM_IMAGES, img_size=IMAGE_SIZE, mri_type="FLAIR", split="train",
                               rotate=0):

        files = sorted(glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"),
                       key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        middle = len(files) // 2
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        img3d = np.stack([self.__load_dicom_image(f, rotate=rotate) for f in files[p1:p2]]).T
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        if np.min(img3d) < np.max(img3d):
            img3d = img3d - np.min(img3d)
            img3d = img3d / np.max(img3d)

        return np.expand_dims(img3d, 0)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __get_data(self, data):
        data = np.array(data)
        images = []
        X = []
        Y = []
        for id in data:
            images.append([self.__load_dicom_images_3d(scan_id=id[0]), id[1]])
        for img in images:
            X.append(img[0])
            Y.append(img[1])
        Y = list(map(int,Y))
        return np.array(X), np.array(Y)

    def __getitem__(self, index):
        print(index)
        data = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__get_data(data)
       
        return x, y

class MultiBranchCNN(tf.keras.Model):
    def __init__(self):
        super(MultiBranchCNN,self).__init__()
        # self.inputA = tf.keras.Input(shape=(1,256,256,64))

        self.conv3d = Conv3D(64, input_shape=(1,256,256,64),kernel_size=(3, 3,3), activation='relu', padding='same')
        self.maxpool3d = MaxPool3D(pool_size=(3,3, 3))
        self.conv3d2 = Conv3D(64, kernel_size=(3,3, 3), activation='relu', padding='same')
        self.maxpool3d2 = MaxPool3D(pool_size=(3,3 ,3))
        self.conv3d3 = Conv3D(64, kernel_size=(3,3, 3), activation='relu', padding='same')
        self.maxpool3d3 = MaxPool3D(pool_size=(3,3, 3))
        self.Flatten = Flatten()
        self.Dense = Dense(512, activation='relu')
        self.Dropout = Dropout(0.1)
        self.Dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        print(type(inputs))
        # x = self.inputA(inputs)
        x = self.conv3d(inputs)
        x = self.maxpool3d(x)
        x = self.conv3d2(x)
        x = self.maxpool3d2(x)
        x = self.conv3d3(x)
        x = self.maxpool3d3(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Dropout(x)
        x = self.Dense2(x)
        return x

def __getitem__(self, index):
       
        data = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__get_data(data)
        print(x.shape , y.shape , index)
       
        return x, y

def call(self, inputs):
        print(type(inputs))
        print(inputs)
        # x = self.inputA(inputs)
        x = self.conv3d(inputs)
        
        x = self.maxpool3d(x)
        
        x = self.conv3d2(x)
        
        x = self.maxpool3d2(x)
      
        x = self.conv3d3(x)
      
        x = self.maxpool3d3(x)
       
        x = self.Flatten(x)
       
        x = self.Dense(x)
       
        x = self.Dropout(x)
       
        return self.Dense2(x)