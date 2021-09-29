from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import Center_Crop, Compose


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        # self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
        val_data_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/3D-RU-Net/Data/Valid'
        self.filename_list = [os.path.join(val_data_path, ID) for ID in os.listdir(val_data_path)]

        self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)]) 

    def znorm(self, data):
        data = data.astype(np.float64)
        mean, std = data.mean(), data.std()
        if std == 0:
            return None
        data -= mean
        data /= std
        return data

    def __getitem__(self, index):

        # ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        # seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
        ct = sitk.ReadImage(os.path.join(self.filename_list[index], 'HighRes', 'Image.nii'))
        seg = sitk.ReadImage(os.path.join(self.filename_list[index], 'HighRes', 'Label.nii'), sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array = (ct_array-np.min(ct_array))/(np.max(ct_array)-np.min(ct_array))
        seg_array = sitk.GetArrayFromImage(seg)

        if ct_array.shape[1] != 256 or ct_array.shape[2] != 256:
            ct_array = zoom(ct_array, (1, 256/ct_array.shape[1], 256/ct_array.shape[2]), order=3)
            seg_array = zoom(seg_array, (1, 256 / seg_array.shape[1], 256 / seg_array.shape[2]), order=0)

        # ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        # ct_array = self.znorm(ct_array, seg_array)

        if self.transforms:
            ct_array,seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Dataset(args, mode='train')

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())