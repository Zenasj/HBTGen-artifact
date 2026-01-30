import torch

import pandas as pd
import numpy as np
from Dataset.BaseDataset import BaseDataset

from torch.utils.data import Dataset
class SpineDataSet(Dataset):

    def __init__(self, params, df, spine_main_path, seg_main_path, isTrain=False):
        """
        Parameters
        ----------
        params : Parameters.ParameterBuilder.Parameters
            Parameters for model, training and dataset
        df : pd.Dataframe
            Dataframe used to track samples
        spine_main_path : str
            String for the location of the samples
        seg_main_path : str
            String for the location of the labels
        """
        self.df = df
        self.params = params
        self.isTrain = isTrain
        self.spine_main_path = spine_main_path
        self.seg_main_path = seg_main_path

        self.seg_filename_list = df['seg_filename'].to_list()
        self.spine_filename_list = df['spine_filename'].to_list()

        from ImageUtils.CropPadImg import CropPadImg

        self.crop_pad = CropPadImg(sample_size=self.params.sample_size,
                                   translation=False,
                                   centered=True)

    def __len__(self):
        """
        Length of total number of samples
        """
        return len(self.seg_filename_list)

    def __getitem__(self, sample_idx):
        """
        Generates a single sample
        """
        spine_sample = self.spine_filename_list[sample_idx]
        seg_sample = self.seg_filename_list[sample_idx]

        x_data = np.load(self.spine_main_path + '/' + spine_sample).astype(np.float32)
        y_data = np.load(self.seg_main_path + '/' + seg_sample).astype(np.uint8)

        x_data = np.where(x_data < -1024, -1024, x_data)
        x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min()) * 255
        x_data, y_data = self.crop_pad([x_data, y_data])

        x_data = np.expand_dims(x_data, axis=0)
        y_data = np.expand_dims(y_data, axis=0)

        x_data = x_data.astype(np.float32)
        y_data = y_data.astype(np.float32)

        sample = {'image': x_data,
                  'label': y_data}
        return sample