import numpy as np

import torch
print(torch.version.cuda) # 11.0
print(torch.__version__) # 1.7.0

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
class Images_Dataset(torch.utils.data.Dataset):

    def __init__(self, path_lmdb, path_lmdb_key_pkl, transformI=None, transformM=None):
        self.path_lmdb_key_pkl = path_lmdb_key_pkl
        self.path_lmdb = path_lmdb

        self.list_key_read = []
        with open(self.path_lmdb_key_pkl, 'rb') as f:
            self.list_key_read = pickle.load(f)

        self.lmdb_env = lmdb.open(self.path_lmdb)
        self.datum = caffe_pb2.Datum()

        self.list_pos = generate_x_y()

        self.Data_get = mod.get_function("Data_get")
        self.SpatialAugmentation = mod10.get_function("SpatialAugmentation")

    def __len__(self):
        return len(self.list_key_read)

    def __getitem__(self, idx):
        self.datum.ParseFromString(value)
        data = caffe.io.datum_to_array(self.datum)
        data_img = data[:3, ...]  # [3,540,1760]
        data_label = data[3:5, ...]  # [2,540,1760]


        loc_gt = np.zeros([16,32,96]).astype(np.float32)
        conf_gt = np.zeros([8,32,96]).astype(np.float32)

        img_out = np.zeros([3, 128, 384]).astype(np.float32)

        self.Data_get(cuda.In(data_label.astype(np.float32)), cuda.Out(loc_gt),
                        cuda.Out(conf_gt), block=(12, 8, 1), grid=(8, 4, 1))

        self.SpatialAugmentation(cuda.In(data_img.astype(np.float32)),
                              cuda.Out(img_out), block=(384, 1, 1), grid=(3, 1, 128))
        return data, img_out, loc_gt, conf_gt





if __name__ == "__main__":
    path_lmdb_key_pkl = "/media/lmdb_key_big.pkl"
    path_lmdb = "/media/train/data_lmdb/"
    batch_size = 2
    dataset_lmdb = Images_Dataset(path_lmdb, path_lmdb_key_pkl)

    loader = torch.utils.data.DataLoader(dataset_lmdb, batch_size=batch_size, num_workers=1)##num_workers=0
    for i, data_ in enumerate(loader):
        print("***"*3)
        data = data_[0][0].numpy()