from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
import csv
import random
import pandas as pd

class Dataset_Union_ALL(Dataset): 
    def __init__(self, paths, mode='train', image_size=128, 
                 transform=None, threshold=500, pcc=False):
        self.paths = paths
        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        assert mode in ['train','val','test']
        self.pcc = pcc
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if(self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if(len(random_index)>=1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                                affine=subject.label.affine),
                                    image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask', 
                                        target_shape=(self.image_size,self.image_size,self.image_size))(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.mode == "train":
            return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
 
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f'labels{self.mode}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    label_path = os.path.join(path, f'labels{self.mode}', f'{base}.nii.gz')
                    self.image_paths.append(label_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)

class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Dataset_CSV(Dataset_Union_ALL):
    def __init__(self, csv_folder, mode='train', image_size=128, 
                 transform=None, expert='digestion',
                 threshold=500, pcc=False):
        self.csv_folder = csv_folder
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        assert mode in ['train','val','test']
        self.pcc = pcc
        self.expert = expert
        self._set_file_paths()

    def _read_csv(self,file_path):
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)[1:]  # 跳过第一行，即表头行
        return data
    
    def _set_file_paths(self):
        self.image_paths = []
        self.label_paths = []
        if self.expert == 'all':
            csv_source = os.path.join(self.csv_folder,f'{self.mode}')
            csv_list = os.listdir(csv_source)
            csv_list = [os.path.join(csv_source,item) for item in csv_list if item.endswith('.csv')]
            for c in csv_list:
                paths = self._read_csv(c)
                for path in paths:
                    self.image_paths.append(path[1])
                    self.label_paths.append(path[2])
        else:
            csv_path = os.path.join(self.csv_folder,f'{self.mode}',f'{self.expert}.csv')
            paths = self._read_csv(csv_path)
            for path in paths:
                self.image_paths.append(path[1])
                self.label_paths.append(path[2])

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])
        
        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())
        #
        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if(self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if(len(random_index)>=1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                                affine=subject.label.affine),
                                    image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask', 
                                        target_shape=(self.image_size,self.image_size,self.image_size))(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.mode == "train":
            return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]

if __name__ == "__main__":
    NotImplemented
