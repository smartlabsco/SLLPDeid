import cv2
import os
import numpy as np
from tqdm import tqdm
import time
from utils.helpers import *
from utils.data_loaders import *

from shapely.geometry import Polygon


def compute_iou(b1, b2):
    b1 = b1.squeeze()
    b2 = b2.squeeze()

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
import random

def inference(network, eval_dataloader, device, save_dir, save_img):
    start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    print("evaltarr")
    print(eval_dataloader)
    i = 0
    j = 0

    with torch.no_grad():

        network.eval()

        for idx, data in enumerate(tqdm(eval_dataloader)): ## 여기서 에러?

            img, img_t, label = data
            #print("label is?")
            #print(img.shape)
            img = img.squeeze().numpy()
    
            [img_t, label] = img2cuda([img_t, label], device)
            #print(img_t.shape)

            pred = network(img_t)


            pred = pred.detach().cpu().numpy().squeeze()

            h, w, c = img.shape
            pred[0::2] = pred[0::2] * w
            pred[1::2] = pred[1::2] * h

            pred = pred.astype('int32')
            #print("predicted 좌표")
            #print(pred)

            
            if save_img:
                img2 = img.copy()
                #cv2.line(img2, (pred[0], pred[1]), (pred[0], pred[1]), (0, 0, 255), thickness=10) ## 실제로는 점이 찍힘
                #cv2.line(img2, (pred[2], pred[3]), (pred[2], pred[3]), (0, 0, 255), thickness=10)
                #cv2.line(img2, (pred[4], pred[5]), (pred[4], pred[5]), (0, 0, 255), thickness=10)
                #cv2.line(img2, (pred[6], pred[7]), (pred[6], pred[7]), (0, 0, 255), thickness=10)
                
                cv2.imwrite(f'{save_dir}/test{idx}_origin.jpg', img2)
                


                
                png_img = cv2.imread('bunhogya3.png', cv2.IMREAD_UNCHANGED)

                # pred에서 좌표를 가져옵니다.
                pts_src = np.array([[0, 0], [0, png_img.shape[0]], [png_img.shape[1], png_img.shape[0]], [png_img.shape[1], 0]], dtype=np.float32)
                pts_dst = np.array([pred[0:2], pred[2:4], pred[4:6], pred[6:8]], dtype=np.float32)

                # 원본 png 이미지와 대상 img 이미지 사이의 변환 행렬을 얻습니다.
                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

                # png 이미지를 대상 img 이미지 좌표로 변환합니다.
                warped_img = cv2.warpPerspective(png_img, matrix, (img.shape[1], img.shape[0]))

                # 변환된 png 이미지를 원래의 img 이미지에 병합합니다.
                mask = (warped_img[:, :, 3] != 0)  # png 이미지의 알파 채널이 0이 아닌 위치에 대한 마스크를 생성합니다.
                warped_img_rgb = warped_img[:, :, :3]
                img[mask] = warped_img_rgb[mask]

                #print(img.shape)


                img_tensor = torch.Tensor(img).permute(2, 0, 1)  # (channels, height, width)

                # 이미지 크기 조정
                resize_transform = transforms.Compose([
                    transforms.ToPILImage(),  # 텐서를 PIL 이미지로 변환
                    transforms.Resize((128, 256)),  # 높이 128, 너비 256으로 조정
                    transforms.ToTensor()  # 다시 텐서로 변환
                ])

                # 크기 조정 적용
                img_resized = resize_transform(img_tensor)

                # 배치 차원 추가
                img_batch = img_resized.unsqueeze(0)  # (1, channels, height, width)

                #print(img_batch.shape)
                img_batch = img_batch.to(device)


                pred2 = network(img_batch)


                pred2 = pred2.detach().cpu().numpy().squeeze()

                h, w, c = img.shape
                pred2[0::2] = pred2[0::2] * w
                pred2[1::2] = pred2[1::2] * h

                pred2 = pred2.astype('int32')


                #print("pred and 2")
                #print(pred)
                #print(pred2)
                #print(compute_iou(pred, pred2))
                iou = compute_iou(pred, pred2)
                i = i + iou
                if iou > 0.75:
                    j = j+1



                cv2.imwrite(f'{save_dir}/test{idx}.jpg', img)
        print("IoU result::")
        print(i/200)
        
        print("소요시간")
        end_time = time.time() - start_time
        print(f"Elapsed time: {end_time:.3f} seconds")
        print("FPS is :")
        print(f"FPS: {200/end_time:.3f} ")
        print("j is")
        print(j)



    

