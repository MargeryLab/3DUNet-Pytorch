from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk

class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels

        self.ct = sitk.ReadImage(data_path)
        self.data_np = sitk.GetArrayFromImage(self.ct)  #(95,256,256)
        self.ori_shape = self.data_np.shape

        # if self.data_np.shape[1] == 384 and self.data_np.shapeshape[2]==492:
        #     xy_down_scale = 0.666666666666666666
        #     self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, xy_down_scale, xy_down_scale), order=3)
        # elif self.data_np.shape[1] != self.data_np.shape[2]:
        #     xy_down_scale = 0.8
        #     self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, xy_down_scale, xy_down_scale), order=3)
        # else:
        #     self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3) # 双三次重采样(95,128,128)
        # self.data_np[self.data_np > args.upper] = args.upper
        # self.data_np[self.data_np < args.lower] = args.lower
        self.data_np = self.znorm(self.data_np)    #200 （95，128，128）
        self.resized_shape = self.data_np.shape
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)    #（96，128，128）48,24
        self.padding_shape = self.data_np.shape #(96,128,128)
        # 对数据按步长进行分patch操作，以防止显存溢出
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)#（3，48，128，128）

        # 读取一个label文件 shape:[s,h,w]
        self.seg = sitk.ReadImage(label_path,sitk.sitkInt8)
        self.label_np = sitk.GetArrayFromImage(self.seg)    #（95，256，256）
        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long() #（1，95，256，256）

        # 预测结果保存
        self.result = None

    def znorm(self, data):
        data = data.astype(np.float64)
        mean, std = data.mean(), data.std()
        if std == 0:
            return None
        data -= mean
        data /= std
        return data

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])    #[48, 128, 128]
        data = torch.FloatTensor(data).unsqueeze(0) #torch.Size([1, 48, 128, 128])
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):

        patch_s = self.result.shape[2]  #48

        N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1    #3
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1],self.ori_shape[2]))  #[3, 96, 256, 256]itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]#torch.Size([3, 96, 256, 256])
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1#torch.Size([3, 96, 256, 256])

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum    #torch.Size([3, 96, 256, 256])
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]#torch.Size([3, 95, 256, 256])
        return img.unsqueeze(0)

    def padding_img(self, img, size, stride):
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((s, img_h, img_w),dtype=np.float32)
        tmp_full_imgs[:img_s] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape #（96，128，128）
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1    #3

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, size, img_h, img_w), dtype=np.float32)   #(3,48,128,128)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[s * stride : s * stride + size] #(48, 128, 128)
            patches[s] = patch

        return patches  # array with all the full_imgs divided in patches #(3,48,128,128)

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

def Test_Datasets(args):
    test_data_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/3D-RU-Net/Data_256/Test'
    filename_list = [os.path.join(test_data_path, ID) for ID in os.listdir(test_data_path)]

    print("The number of test samples is: ", len(filename_list))
    for file in filename_list:
        print("\nStart Evaluate: ", file[0])
        yield Img_DataSet(os.path.join(test_data_path,file, 'HighRes', 'Image.nii'),
                          os.path.join(test_data_path,file, 'HighRes', 'Label.nii'),args=args), file
