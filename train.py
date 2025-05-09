#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse
from tqdm import tqdm
import json

respth = './res_distributed'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()

# Create checkpoints directory
checkpoint_dir = osp.join(respth, 'checkpoints')
if not osp.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    parse.add_argument(
            '--save_interval',
            dest = 'save_interval',
            type = int,
            default = 5000,
            help = 'Interval to save model checkpoints'
            )
    parse.add_argument(
            '--eval_interval',
            dest = 'eval_interval',
            type = int,
            default = 1000,
            help = 'Interval to evaluate model'
            )
    return parse.parse_args()

def save_checkpoint(net, optimizer, epoch, iteration, loss, is_best=False, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
        'optimizer_state_dict': optimizer.optim.state_dict(),
        'optimizer_it': optimizer.it,
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = osp.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if it's the best so far
    if is_best:
        best_model_path = osp.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        logger.info(f'Saved best model with loss: {loss:.4f}')

def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.optim.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.it = checkpoint['optimizer_it']
    return checkpoint['epoch'], checkpoint['iteration'], checkpoint['loss']

def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    setup_logger(respth)

    # dataset
    n_classes = 10
    n_img_per_gpu = 16
    n_workers = 8
    cropsize = [448, 448]
    data_root = '/workspace/bisenet_training/complex_seg_bisenet'

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    sampler = None
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    sampler = sampler,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    ## train loop
    msg_iter = 500
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    best_loss = float('inf')
    
    # Training progress bar
    pbar = tqdm(total=max_iter, desc='Training Progress')
    
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        current_loss = loss.item()

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'epoch': epoch})

        # Save checkpoint
        if (it + 1) % args.save_interval == 0:
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            save_checkpoint(
                net, 
                optim, 
                epoch, 
                it, 
                current_loss, 
                is_best=is_best,
                filename=f'checkpoint_{it+1}.pth'
            )

        # Evaluate model
        if (it + 1) % args.eval_interval == 0:
            net.eval()
            with torch.no_grad():
                # Add your evaluation code here
                pass
            net.train()

        #  print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed

    pbar.close()
    
    # Save final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))

if __name__ == "__main__":
    train()
