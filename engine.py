# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys,numpy
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import random
#from tsne import tSNE

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    RandAug_flag=True,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    #     samples = samples.to(device, non_blocking=True)
    #     targets = targets.to(device, non_blocking=True)


    # for it, ((data, jig_l, class_l), d_idx) in enumerate(data_loader):
    #     samples, jig_l, targets, d_idx = data.to(device), jig_l.to(device), class_l.to(device), d_idx.to(device)


    for it, ((data, data_randaug, class_l, domain_l), d_idx) in enumerate(data_loader):
        data, data_randaug, class_l, domain_l, d_idx = data.to(device), data_randaug.to(device), \
                                                       class_l.to(device), domain_l.to(device), \
                                                       d_idx.to(device)

        if RandAug_flag:
            samples = torch.cat((data, data_randaug))
            targets = torch.cat((class_l, class_l))
        else:
            samples = data
            targets = class_l

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
    
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, test, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # for images, target in metric_logger.log_every(data_loader, 10, header):
    #     images = images.to(device, non_blocking=True)
    #     target = target.to(device, non_blocking=True)

    # for it, ((data, nouse, class_l), _) in enumerate(data_loader):
    #     images, nouse, target = data.to(device), nouse.to(device), class_l.to(device)

    for it, ((data, _, class_l, domain_l), _) in enumerate(data_loader):
        images, target = data.to(device), class_l.to(device)   #增广用


        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
