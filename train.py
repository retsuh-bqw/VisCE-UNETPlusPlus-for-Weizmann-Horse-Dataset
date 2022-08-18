import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from test import test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import HorseDataset
from model import UnetPP
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from utils import calculate_iou, calc_boundary_iou, BCE_DiceCoef_loss, mask_to_boundary
logger = logging.getLogger(__name__)



def train(model, train_dataloader, test_dataloader, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    

    logger.info('epoch \t mean_IOU \t\t mean_BIOU \t\t loss')
    deep_supervision = False
    for epoch in range(1, args.epoch):
        model.train()
        
        IOU = []
        B_IOU = []
        LOSS = []

        with tqdm(train_dataloader) as loader:
            for image_batch, mask_batch in loader:
                loader.set_description(f"Epoch {epoch}")
                image_batch = image_batch.cuda()
                mask_batch = mask_batch.cuda()

                output = model(image_batch)
                loss = BCE_DiceCoef_loss(output, mask_batch)
                iou = calculate_iou(output.squeeze(dim=1), mask_batch)
                b_iou = calc_boundary_iou(mask_batch, output.squeeze(dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                LOSS.append(loss)
                IOU.append(iou)
                B_IOU.append(b_iou)
                loader.set_postfix(loss=loss.item(), IOU='{:.3f}'.format(iou))
        scheduler.step()
        mean_iou = sum(IOU) / len(IOU)
        mean_b_iou = sum(B_IOU) / len(B_IOU)
        mean_loss = sum(LOSS) / len(LOSS)

        logger.info('%d \t\t %.4f \t\t %.4f \t\t %.4f',
            epoch, mean_iou, mean_b_iou, mean_loss)

        if epoch % 5 == 0:
            test_iou, test_biou, test_loss = test(model, test_dataloader)

            logger.info('%s \t\t %.4f \t\t %.4f \t\t %.4f',
            'TEST', test_iou, test_biou, test_loss)


    saved_name = '{0}-mIOU{1:.4f}-BIOU{2:.4f}.pt'.format('UnetPP', mean_iou, mean_b_iou)
    torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='saved weight path')
    parser.add_argument('--out_dir', type=str, default='./log/', help='log output dir')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--data_root', type=str, default='./weizmann_horse_db', help='dataset path')
    parser.add_argument('--epoch', type=int, default=100, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--workers', type=int, default=8, help='num of workers for data retrieval')

    args = parser.parse_args()

    logfile = os.path.join(args.out_dir, 'UNET-D{0}-H{1}-{2}.log'.format(time.localtime(time.time())[2],
                                                                time.localtime(time.time())[3],
                                                                time.localtime(time.time())[4]))
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    idx = np.arange(327)
    np.random.shuffle(idx)

    training_idx = idx[:int(327 * 0.85)]
    testing_idx = idx[int(327 * 0.85):]
    train_transforms = Transform_Compose([Train_Transform(image_size=224), Totensor()])
    test_transforms = Transform_Compose([Test_Transform(image_size=224), Totensor()])
    train_data = HorseDataset(args.data_root, training_idx, train_transforms)
    test_data = HorseDataset(args.data_root, testing_idx, test_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True
    )



    model = UnetPP().to(device)
    # model.load_state_dict(torch.load(args.pretrained))

    train(model, train_dataloader, test_dataloader, args)