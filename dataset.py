import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128), img_type = 'GS'):
        self.phase = phase
        self.input_shape = input_shape
        self.img_type = img_type

        with open(os.path.join(root, data_list_file), 'r') as fd:
            imgs = fd.readlines()

        # self.class_to_idx = [None] * (max(int(line.split()[1]) for line in imgs) + 1)  # Ensure array size fits max index
        # for line in imgs:
        #     class_name, index = line.split()[0].split('/')[0], int(line.split()[1])
        #     if self.class_to_idx[index] is None:
        #         self.class_to_idx[index] = class_name

        self.class_to_idx = {}
        for line in imgs:
            class_name, index = line.split()[0].split('/')[0], int(line.split()[1])
            self.class_to_idx[class_name] = index

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        # RGB version
        if self.img_type == 'RGB':
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

        elif self.img_type == 'GS':
            # grayscale version
            normalize = T.Normalize(mean=[0.5], std=[0.5])

        else:
            print('Dataset: img_type invalid {img_type}')
        
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(128),
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(128),
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        if self.img_type == 'GS':
            # grayscale version
            data = data.convert('L')
        elif self.img_type == 'RGB':
            # RGB version
            data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)




if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)