# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    print('engine.py - Entering train_one_epoch')
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    iter__ = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
           
        '''
        targets.keys() -> ['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']
        targets['orig_size'] has the original size of images. But targets['size'] will be the one that will be used.
        Let us say, we have batch_size = 2 having size : tensor([544, 694]) and size : tensor([544, 817]). Then,
        samples_size will be ([2, 3, 544, 817]).
        Similarly for size : tensor([852, 640]) and size : tensor([648, 640]), samples_size will be ([2, 3, 852, 640]).
        After passing through backbone - in this case resnet-50 of stride 32 - samples_size will get reduced to [27, 20]
        '''

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        
        
        print_flag = 0
        if (iter__ == 0 or iter__ == 55) and epoch == 0:
            print('-----------------------------------')
            print_flag = 1       
        if print_flag:
            sz = samples.tensors.shape
            print(f"Engine - targets[0].keys() : {targets[0].keys()}")
            print(f"Engine - targets[0] Num boxes: {targets[0]['boxes'].size()}, orig_size : {targets[0]['orig_size']}, size : {targets[0]['size']}")
            print(f"Engine - targets[1] Num boxes: {targets[1]['boxes'].size()}, orig_size : {targets[1]['orig_size']}, size : {targets[1]['size']}")
            print(f'Engine - iter : {iter__}, samples.size() : {sz}, Resnet-stride 32 reshapes to: [{sz[0]},{sz[1]},{math.ceil(sz[2]/32)},{math.ceil(sz[3]/32)}]')
        iter__ += 1
            
        outputs = model(samples, print_flag)
        if print_flag:
            print(f'Engine - outputs : {outputs.keys()}')
            print(f"Engine - pred_logits: {outputs['pred_logits'][0].size()}, pred_boxes: {outputs['pred_boxes'][0].size()}, aux_o/p type: {type(outputs['aux_outputs'])}")
        loss_dict = criterion(outputs, targets, print_flag)
        if print_flag:
            print('Engine - loss_dict:', loss_dict)
            print('----------------------------------')
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    print('engine.py - evaluate')
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    iter__ = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
                
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        
        
        print(f'Eval Iteration : {iter__}, samples.size() : {samples.tensors.shape}')
        iter__ +=1 
        print_flag = 0
        
        outputs = model(samples, print_flag)
        loss_dict = criterion(outputs, targets, print_flag)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
