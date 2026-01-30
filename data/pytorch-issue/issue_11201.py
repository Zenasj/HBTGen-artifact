import torch
import numpy as np

class PhysicsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.gt_spectra = list(self.data_dir.glob("*.npz"))
        self.gt_parameters = json.load(
            open(self.data_dir / "all_params.json", 'r'))

    def __len__(self):
        return len(self.gt_spectra)

    def __getitem__(self, index):
        with np.load(self.gt_spectra[index]) as data:
            pdata = data['spectrum']
        pdata = (pdata - pdata.min()) / (pdata.max() - pdata.min())
        pdata = torch.from_numpy(pdata).float()
        parameters = self.gt_parameters[self.gt_spectra[index].name.replace(
            ".npz", "")]
        if self.transform:
            pdata = self.transform(pdata)

        # create output tensor with normalised weights
        gt_tensor = torch.from_numpy(
            np.asarray([(parameters[k] - KEYS[k]['min']) /
                        (KEYS[k]['max'] - KEYS[k]['min'])
                        for k in KEYS])).float()
        return {
            "spectrum": pdata,
            "gt_tensor": gt_tensor,
            "filename": self.gt_spectra[index].name
        }