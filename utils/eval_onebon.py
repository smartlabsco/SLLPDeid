import cv2
import os
from tqdm import tqdm

from utils.helpers import *
from utils.data_loaders import *

from shapely.geometry import Polygon


def compute_iou(b1, b2):
    b1 = b1.detach().cpu().numpy().squeeze()
    b2 = b2.detach().cpu().numpy().squeeze()

    box1 = Polygon([(b1[0], b1[1]), (b1[2], b1[3]), (b1[4], b1[5]), (b1[6], b1[7])])
    box2 = Polygon([(b2[0], b2[1]), (b2[2], b2[3]), (b2[4], b2[5]), (b2[6], b2[7])])

    if not box1.is_valid:
        box1 = box1.buffer(0)
        return 0
    if not box2.is_valid:
        box2 = box2.buffer(0)
        return 0

    intersect = box1.intersection(box2).area
    union = box1.union(box2).area
    iou = intersect / union

    return iou


def evaluate(network, eval_dataloader, device, logging):

    with torch.no_grad():

        network.eval()

        eval_metric = 0

        for idx, data in enumerate(tqdm(eval_dataloader)):

            img, label = data
            [img, label] = img2cuda([img, label], device)

            pred = network(img)

            eval_metric += compute_iou(pred, label)

        eval_metric /= len(eval_dataloader)

        print(eval_metric)

        return eval_metric


def evaluate_v(network, eval_dataloader, device, save_dir, save_img):

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():

        network.eval()
        avg_iou = 0

        for idx, data in enumerate(tqdm(eval_dataloader)):

            img, img_t, label = data
            img = img.squeeze().numpy()
            
            [img_t, label] = img2cuda([img_t, label], device)

            pred = network(img_t)

            iou_img = compute_iou(pred,label)
            avg_iou += iou_img

            pred = pred.detach().cpu().numpy().squeeze()
            label = label.detach().cpu().numpy().squeeze()

            h, w, c = img.shape
            pred[0::2] = pred[0::2] * w
            label[0::2] = label[0::2] * w
            pred[1::2] = pred[1::2] * h
            label[1::2] = label[1::2] * h

            label = label.astype('int32')
            pred = pred.astype('int32')

            
            if save_img:
                cv2.line(img, (label[0], label[1]), (label[0], label[1]), (0, 255, 0), thickness=4)
                cv2.line(img, (label[2], label[3]), (label[2], label[3]), (0, 255, 0), thickness=4)
                cv2.line(img, (label[4], label[5]), (label[4], label[5]), (0, 255, 0), thickness=4)
                cv2.line(img, (label[6], label[7]), (label[6], label[7]), (0, 255, 0), thickness=4)
                cv2.line(img, (pred[0], pred[1]), (pred[0], pred[1]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[2], pred[3]), (pred[2], pred[3]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[4], pred[5]), (pred[4], pred[5]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[6], pred[7]), (pred[6], pred[7]), (0, 0, 255), thickness=4)
                
                cv2.imwrite(f'{save_dir}/{iou_img:.3f}_test{idx}.jpg', img)
            

        avg_iou /= len(eval_dataloader)
        print(f"IOU RESULTS : {avg_iou}")


def evaluate(network, eval_dataloader, device, logging):

    with torch.no_grad():

        network.eval()

        eval_metric = 0

        for idx, data in enumerate(tqdm(eval_dataloader)):

            img, label = data
            [img, label] = img2cuda([img, label], device)

            pred = network(img)

            eval_metric += compute_iou(pred, label)

        eval_metric /= len(eval_dataloader)

        print(eval_metric)

        return eval_metric


def inference(network, eval_dataloader, device, save_dir, save_img):

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():

        network.eval()

        for idx, data in enumerate(tqdm(eval_dataloader)):

            img, img_t, label = data
            img = img.squeeze().numpy()
            
            [img_t, label] = img2cuda([img_t, label], device)

            pred = network(img_t)


            pred = pred.detach().cpu().numpy().squeeze()

            h, w, c = img.shape
            pred[0::2] = pred[0::2] * w
            pred[1::2] = pred[1::2] * h

            pred = pred.astype('int32')

            
            if save_img:
                cv2.line(img, (pred[0], pred[1]), (pred[0], pred[1]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[2], pred[3]), (pred[2], pred[3]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[4], pred[5]), (pred[4], pred[5]), (0, 0, 255), thickness=4)
                cv2.line(img, (pred[6], pred[7]), (pred[6], pred[7]), (0, 0, 255), thickness=4)
                
                cv2.imwrite(f'{save_dir}/test{idx}.jpg', img)