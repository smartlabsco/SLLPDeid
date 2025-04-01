import os

import torch
from torch.utils.data import DataLoader

from config import parse_args
from model import LPDetectionNet
from utils.data_loaders import Visual_Dataset
from utils.helpers import *
from utils.eval import *

def visualize():
    
    args = parse_args()

    GPU_NUM = args.gpu_num
    NUM_WORKERS = args.num_workers

    EXP_DIR = os.path.join('experiments', args.exp_name)
    WEIGHTS = args.weights

    SAVE_DIR = os.path.join(args.save_dir, args.exp_name)
    SAVE_IMG = args.save_img


    # Set up Dataset
    test_dataset = Visual_Dataset(args, 'test')

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # Set up Network
    network = LPDetectionNet(args)


    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network.to(device)


    # ckpt = torch.load(os.path.join('experiments', 'exp_4', 'ckpt.pth'))
    ckpt = torch.load(os.path.join(EXP_DIR, WEIGHTS))
    network.load_state_dict(ckpt['network'])

    with torch.no_grad():
        REF_IMG_PATH = 'bunhogya3.png'
        #inference_with_pasting(network, test_dataloader, device, REF_IMG_PATH, SAVE_DIR, SAVE_IMG)
        inference(network, test_dataloader, device, save_dir = SAVE_DIR, save_img = SAVE_IMG)


if __name__ == '__main__':
    visualize()    