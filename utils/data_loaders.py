import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import parse_args
from utils.helpers import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')


        self.args = args
        self.imgs = []
        self.labels = []
        self.subset = subset
        self.image_size = args.image_size

        for file in os.listdir(os.path.join(args.data_dir, args.dataset_img, subset)):
            if file.endswith(('.jpg', '.png', '.PNG')):
                self.imgs.append(file)

        for file in os.listdir(os.path.join(args.data_dir, args.dataset_label, subset)):
            if file.endswith('.json'):
                self.labels.append(file)

        self.imgs = sorted(self.imgs)
        self.labels = sorted(self.labels)

        self.transform_img = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize([int(self.image_size[0]), int(self.image_size[1])]),
            ]
        )

        self.transform_label = transforms.Compose(
            [transforms.ToTensor()]
        )


    def __getitem__(self, idx):

        DATA_DIR        = self.args.data_dir
        DATASET_IMG     = self.args.dataset_img
        DATASET_LABEL   = self.args.dataset_label

        # Read in Image
        imgpath = os.path.join(DATA_DIR, DATASET_IMG, self.subset, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)

        labelpath = os.path.join(DATA_DIR, DATASET_LABEL, self.subset, self.labels[idx])

        label = list()
        with open(labelpath, 'r') as json_file:
            jf = json.load(json_file)
            for _, value in jf.items():
                label.append(value[0] / img.shape[1])
                label.append(value[1] / img.shape[0])

        img = self.transform_img(img)
        label = np.array(label, dtype = np.float32)

        return img, label
    
    def __len__(self):
        return len(self.imgs)


class Visual_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.imgs = []
        self.labels = []
        self.subset = subset
        self.image_size = args.image_size

        for file in os.listdir(os.path.join(args.data_dir, args.dataset_img, subset)):
            if file.endswith(('.jpg', '.png', '.PNG')):
                self.imgs.append(file)

        self.imgs = sorted(self.imgs)

        os.path.join(args.data_dir, args.dataset_label, subset)

        if os.path.exists(os.path.join(args.data_dir, args.dataset_label, subset)):
            for file in os.listdir(os.path.join(args.data_dir, args.dataset_label, subset)):
                if file.endswith('.json'):
                    self.labels.append(file)
        else: 
            for i in range(len(self.imgs)):
                self.labels.append('')

        self.labels = sorted(self.labels)

        self.transform_img = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize([int(self.image_size[0]), int(self.image_size[1])]),
            ]
        )

        self.transform_label = transforms.Compose(
            [transforms.ToTensor()]
        )


    def __getitem__(self, idx):

        DATA_DIR        = self.args.data_dir
        DATASET_IMG     = self.args.dataset_img
        DATASET_LABEL   = self.args.dataset_label

        # Read in Image
        imgpath = os.path.join(DATA_DIR, DATASET_IMG, self.subset, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB)

        if self.labels[idx] != '':
            labelpath = os.path.join(DATA_DIR, DATASET_LABEL, self.subset, self.labels[idx])

            label = list()
            with open(labelpath, 'r') as json_file:
                jf = json.load(json_file)
                for _, value in jf.items():
                    label.append(value[0] / img.shape[1])
                    label.append(value[1] / img.shape[0])
            label = np.array(label, dtype = np.float32)
        else:
            label = 0

        img_tensor = self.transform_img(img)

        return img, img_tensor, label
    
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    args = parse_args()

    GPU_NUM = 2
    PLAYGROUND = 'playground/'
    BATCH_SIZE = 8

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(PLAYGROUND, exist_ok=True)

    train_dataset = Dataset(args, "test")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )    

    save_idx = 0

    for i, data in enumerate(train_dataloader):

        img, label = data

        print(label)

        if i == 0:
            sys.exit(1)

