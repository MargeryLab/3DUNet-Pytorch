# -*- coding: utf-8 -*-
"""
@author: HuangyjSJTU
"""
import SimpleITK as sitk
import numpy as np
import sys
import os

sys.path.append('./lib/')
import matplotlib.pyplot as pl
from PIL import Image as Img
# import dicom
import cv2
from skimage import filters
from skimage.measure import label, regionprops

# For intensity normalization

DataRoot = '../Data/'
ManualNormalize = True
ResRate = ['HighRes', 'MidRes', 'LowRes']
# ToSpacing={'HighRes':[1,1,4],'MidRes':[1.5,1.5,4],'LowRes':[2,2,4]}           #160，106，80
ToSpacing = {'HighRes': [0.5, 0.5, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}  # 320,160,106


def ReadImageAndLabel(CasePath, labelPath, ID):
    # Reading Images
    Image = sitk.ReadImage(os.path.join(CasePath, ID))

    Spacing = Image.GetSpacing()  # 两个像素之间的间隔<class 'tuple'>: (0.724609, 0.724609, 5.0)
    Origin = Image.GetOrigin()  # 原始图像中心点在相机坐标系的位置<class 'tuple'>: (-195.0, -185.5, -315.5)
    Direction = Image.GetDirection()  # 读取图像方向，一般一系列图像都是同样的<class 'tuple'>: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Reading Labels
    Label = sitk.ReadImage(os.path.join(labelPath, ID))
    # 19,512,512
    LabelArray = sitk.GetArrayFromImage(Label)
    # print(LabelArray.shape, Spacing)

    Label = sitk.GetImageFromArray(LabelArray)
    Label.SetSpacing(Spacing)  # 对sitk的image1处理完后恢复到世界坐标系
    Label.SetOrigin(Origin)
    Label.SetDirection(Direction)

    return Image, Label


def Resampling(Image, Label):
    Size = Image.GetSize()
    Spacing = Image.GetSpacing()  # <class 'tuple'>: (0.3125, 0.3125, 3.2999441623687744)
    if Spacing[0] > 0.5 and Spacing[0]<0.53:
        ToSpacing = {'HighRes': [0.629, 0.629, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.53:
        ToSpacing = {'HighRes': [0.664, 0.664, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] == 0.5:
        ToSpacing = {'HighRes': [0.625, 0.625, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.49 and Spacing[0] < 0.5:
        ToSpacing = {'HighRes': [0.621, 0.621, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.48 and Spacing[0] < 0.49:
        ToSpacing = {'HighRes': [0.781, 0.781, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.4 and Spacing[0] < 0.42:
        ToSpacing = {'HighRes': [0.666, 0.666, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.39 and Spacing[0] < 0.4:
        ToSpacing = {'HighRes': [0.625, 0.625, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.35 and Spacing[0] < 0.36:
        ToSpacing = {'HighRes': [0.562, 0.562, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.33 and Spacing[0] < 0.34:
        ToSpacing = {'HighRes': [0.531, 0.531, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    elif Spacing[0] > 0.35 and Spacing[0] < 0.36:
        ToSpacing = {'HighRes': [0.437, 0.437, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    # elif Spacing[0] == 410 or Spacing[1] == 410:
    #     ToSpacing = {'HighRes': [0.437, 0.437, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}
    else:
        ToSpacing = {'HighRes': [0.5, 0.5, 2], 'MidRes': [1, 1, 2], 'LowRes': [1.5, 1.5, 2]}

    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    ImagePyramid = []
    LabelPyramid = []
    for i in range(3):
        NewSpacing = ToSpacing[ResRate[i]]
        NewSize = [round(Size[0] * Spacing[0] / NewSpacing[0]), round(Size[1] * Spacing[1] / NewSpacing[1]),
                   round(Size[2] * Spacing[2] / NewSpacing[2])]
        if i==0:
            if NewSize[0] % 2 != 0 or NewSize[1] % 2 != 0:
                print('error')
            print(NewSize)
        Resample = sitk.ResampleImageFilter()
        Resample.SetOutputDirection(Direction)
        Resample.SetOutputOrigin(Origin)
        Resample.SetSize(NewSize)
        Resample.SetInterpolator(sitk.sitkBSpline)
        Resample.SetOutputSpacing(NewSpacing)
        NewImage = Resample.Execute(Image)
        ImagePyramid.append(NewImage)

        Resample = sitk.ResampleImageFilter()
        Resample.SetOutputDirection(Direction)
        Resample.SetOutputOrigin(Origin)
        Resample.SetSize(NewSize)
        Resample.SetOutputSpacing(NewSpacing)
        Resample.SetInterpolator(sitk.sitkNearestNeighbor)
        NewLabel = Resample.Execute(Label)
        LabelPyramid.append(NewLabel)
    return ImagePyramid, LabelPyramid


# We shift the mean value to enhance the darker side
UpperBound = 1.0
LowerBound = -4.0


def Normalization(Image):
    Spacing = Image.GetSpacing()  # <class 'tuple'>: (0.724609, 0.724609, 5.0)
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)
    Array_new = Array.copy()
    Array_new += np.min(Array_new)
    # print(int(Array_new.shape[0]/2- 5))
    Array_new = Array_new[int(Array_new.shape[0] / 2 - 5):int(
        Array_new.shape[0] / 2 + 5)]  # 4:14,<class 'tuple'>: (10, 512, 512)丢弃掉上下4层
    Mask = Array_new.copy()
    for i in range(Array_new.shape[0]):
        otsu = filters.threshold_otsu(Array_new[i])  # "Return threshold value based on Otsu's method.546
        # print(Mask[i][Array_new[i]<0.5*otsu])
        # print(Mask[i])
        Mask[i][Array_new[i] < 0.5 * otsu] = 0
        Mask[i][Array_new[i] >= 0.5 * otsu] = 1
    MaskSave = sitk.GetImageFromArray(Mask)
    MaskSave = sitk.BinaryDilate(MaskSave, 10)  # 膨胀
    MaskSave = sitk.BinaryErode(MaskSave, 10)  # 腐蚀
    Mask = sitk.GetArrayFromImage(MaskSave)

    Avg = np.average(Array[int(Array_new.shape[0] / 2 - 5):int(Array_new.shape[0] / 2 + 5)], weights=Mask)  # 0：10
    Std = np.sqrt(np.average(abs(Array[int(Array_new.shape[0] / 2 - 5):int(Array_new.shape[0] / 2 + 5)] - Avg) ** 2,
                             weights=Mask))
    Array = (Array.astype(np.float32) - Avg) / Std
    Array[Array > UpperBound] = UpperBound
    Array[Array < LowerBound] = LowerBound
    Array = ((Array.astype(np.float64) - np.min(Array)) / (np.max(Array) - np.min(Array)) * 255).astype(
        np.uint8)  # 0-255
    Image = sitk.GetImageFromArray(Array)
    Image.SetDirection(Direction)
    Image.SetOrigin(Origin)
    Image.SetSpacing(Spacing)
    return Image, MaskSave


if __name__ == '__main__':
    PatientNames = os.listdir('MRI')
    PatientNames = sorted(PatientNames)
    for i in range(len(PatientNames)):
        PatientName = PatientNames[i]
        Image, Label = ReadImageAndLabel('MRI', 'label', PatientName)
        # Image, Mask = Normalization(Image)  # image是0-255，Mask是ostu后的Mask
        ImagePyramid, LabelPyramid = Resampling(Image, Label)
        for i in range(len(ResRate)):
            # BodyMask用于区分前景和背景，背景0 前景1
            # sitk.WriteImage(Mask,os.path.join('Normalized', reso, 'body', PatientName+'.nii.gz'))# ostu所提取的mask [512, 512, 10]
            sitk.WriteImage(ImagePyramid[i], os.path.join('fixed_data/Normalized', ResRate[i], 'MRI', PatientName+'.nii.gz'))  # mhd改nii
            sitk.WriteImage(LabelPyramid[i], os.path.join('fixed_data/Normalized', ResRate[i], 'label', PatientName+'.nii.gz'))
