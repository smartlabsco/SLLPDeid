import torch
import torch.nn as nn

from config import parse_args
from utils.helpers import *


class LPDetectionNet(nn.Module):
    def __init__(self, args, num_coordinates = 8, dropout = 0.5):
        super().__init__()
        self.num_coordinates = num_coordinates
        self.image_shape = args.image_size
        self.dropout = dropout
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fclayer = nn.Sequential(
            nn.Linear(64 * int(self.image_shape[0] / 2**4) * int(self.image_shape[1] / 2**4), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_coordinates),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x

if __name__ == '__main__':

    import os

    args = parse_args()

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 4
    H = args.image_size[0]
    W = args.image_size[1]

    x = torch.rand([BATCH_SIZE, 3, H, W]).to(device)

    network = LPDetectionNet(args).to(device)


    pred = network(x)
    print(pred.shape)
