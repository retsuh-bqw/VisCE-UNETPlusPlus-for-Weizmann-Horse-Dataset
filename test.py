import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import logging
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import HorseDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from model import UnetPP
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from utils import calculate_iou, calc_boundary_iou, BCE_DiceCoef_loss, mask_to_boundary
from PIL import Image
from tqdm import tqdm
import time


def test(model, test_data_loader):
    # print("testing...")
    model.eval()
    IOU = []
    B_IOU = []
    LOSS = []
    deep_supervision = False

    with torch.no_grad():
        for test_data, test_mask in test_data_loader:
            test_data = test_data.cuda()
            test_mask = test_mask.cuda()
            loss = 0

            output = model(test_data)
            loss += BCE_DiceCoef_loss(output, test_mask)
            iou = calculate_iou(output.squeeze(dim=1), test_mask)
            b_iou = calc_boundary_iou(test_mask, output.squeeze(dim=1))

            IOU.append(iou)
            B_IOU.append(b_iou)
            LOSS.append(loss)
    mean_iou = sum(IOU) / len(IOU)
    mean_b_iou = sum(B_IOU) / len(B_IOU)
    mean_loss = sum(LOSS) / len(LOSS)

    return mean_iou, mean_b_iou, mean_loss




def show_predict(model, test_data_loader):

    best_model = model.to(device=device)
    best_model.eval()


    cal = 0
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    with torch.no_grad():
        for test_data, test_mask in test_data_loader:
            test_data = test_data.cuda()
            test_mask = test_mask.cuda()
            cal += 1
            B, H, W = test_mask.shape
            for k in range(B):
                pic1 = toPIL(test_data[k])
                pic2 = toPIL(test_mask[k])

                pic1.save(r"./%s/%d_%d_ori.png" % ('demo', cal, k))
                pic2.save(r"./%s/%d_%d_mask.png" % ('demo', cal, k))

            outputs = best_model(test_data)
            # 获得边界
            deep_supervision = False
            mask_boundary = 255 * mask_to_boundary(test_mask, dilation_ratio=0.02, sign=1)
            if deep_supervision is True:
                # print("采用深监督")
                out_boundary = 255 * mask_to_boundary(outputs[-1].squeeze(dim=1), dilation_ratio=0.02, sign=1)
                out = outputs[-1].squeeze(dim=1)
            else:
                # print("无深监督")
                out_boundary = 255 * mask_to_boundary(outputs.squeeze(dim=1), dilation_ratio=0.02, sign=1)
                out = outputs.squeeze(dim=1)

            Save_out = torch.sigmoid(out).data.cpu().numpy()
            Save_out[Save_out > 0.5] = 255

            Save_out[Save_out <= 0.5] = 0
            print(Save_out)
            test_mask_ = torch.sigmoid(test_mask).data.cpu().numpy()
            test_mask_[test_mask_ > 0.5] = 255
            test_mask_[test_mask_ <= 0.5] = 0
            for j in range(B):
                A = Image.fromarray(mask_boundary[j].astype('uint8'))
                B = Image.fromarray(out_boundary[j].astype('uint8'))
                A.save(r"./%s/%d_%d_mask_boundary.png" % ('demo', cal, j))
                B.save(r"./%s/%d_%d_predict_boundary.png" % ('demo', cal, j))
                Y = test_mask_[j].astype('uint8')
                X = Save_out[j].astype('uint8')
                sub = np.abs(X - Y)
                sub = Image.fromarray(sub)
                sub.save(r"./%s/%d_%d_sub.png" % ('demo', cal, j))
                Z = Image.fromarray(X)
                Z.save(r"./%s/%d_%d_predict.png" % ('demo', cal, j))
            if (cal == 1):
                print("预测结束")
                print("-" * 34)
                break

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Model Test')
    parser.add_argument('--weights', type=str, default='./checkpoint/cosWRN34-CIFAR10-PGD.pt', help='saved model path')
    parser.add_argument('--data_root', type=str, default='./weizmann_horse_db', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=8, help='num of workers for data retrieval')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = np.arange(327)
    np.random.shuffle(idx)


    # testing_idx = idx[int(327 * 0.85):]
    test_transforms = Transform_Compose([Test_Transform(image_size=224), Totensor()])
    test_data = HorseDataset(args.data_root, idx, test_transforms)

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )


    model = UnetPP().to(device)
    model.load_state_dict(torch.load(args.weights))

    miou, biou, loss = test(model, test_dataloader)
    print('IOU:{0:.4f}, BIOU:{1:.4f}, Loss:{2:.4f}'.format(miou, biou, loss))

    # show_predict(model, test_dataloader)