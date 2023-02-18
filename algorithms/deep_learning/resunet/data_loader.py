from os.path import join, split
from glob import glob
import skimage.io
import numpy as np

from torch.utils.data import Dataset


class RoadDatasetCUG(Dataset):
    def __init__(self, img_dir, mask_dir):
        super.__init__()
        img_fns = glob(join(img_dir, '*.jpg'))
        mask_fns = []
        for fn in img_fns:
            base_name = split(fn)[-1]
            base_name_mask = base_name.replace('_sat.jpg', '_mask.png')
            mask_fns.sppend(join(mask_dir, base_name_mask))
        self.img_fns = img_fns
        self.mask_fns = mask_fns

    def __getitem__(self, index):
        img_data = skimage.io.imread(self.img_fns[index])  # W*H*C
        img_data = np.moveaxis(img_data, -1, 0)  # C*W*H
        mask_data = skimage.io.imread(self.mask_fns[index], as_gray=True)
        return img_data.astype(np.float32), mask_data.astype(np.float32)

    def __len__(self):
        return len(self.img_fns)
