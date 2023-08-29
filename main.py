# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import sys
import argparse
import datetime
import random
import time
from pathlib import Path
import math
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from typing import Iterable
import util.misc as utils

from models import build_model
from datasets import build_dataset
import datasets.kineticsGEBD as kineticsGEBD
from datasets.kineticsGEBD_eval import eval_GEBD, formattingGEBD, display_table
from datasets.evaluator import general_evaluator

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=kineticsGEBD.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=kineticsGEBD.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        evaluator, _ = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args
        )
        res = evaluator.summarize()
         
        formattingGEBD(res, args.thresh, args.annotation_path)
        
        total_f1 = 0
        test_stats = {}
        eval_thresholds = np.linspace(0.05,0.5,10)      
        for th in eval_thresholds:
            f1_stats = eval_GEBD(args.annotation_path, th)
            test_stats[th] = f1_stats['f1']
            total_f1 += f1_stats['f1']
        test_stats['avg'] = total_f1 / len(eval_thresholds)

        display_table(eval_thresholds, test_stats)

        return

    print("Start training")
    start_time = time.time()

    epoch_list = []
    best_f1_0_05 = 0
    best_f1_avg = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats, train_loss_dict = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args)
        
        lr_scheduler.step()
        if epoch % 1 == 0 and args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth'.format(epoch)]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        evaluator, _ = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args
        )
        res = evaluator.summarize()
        
        formattingGEBD(res, args.thresh, args.annotation_path)
        
        total_f1 = 0
        test_stats = {}
        eval_thresholds = np.linspace(0.05,0.5,10)      
        for th in eval_thresholds:
            f1_stats = eval_GEBD(args.annotation_path, th)
            test_stats[th] = f1_stats['f1']
            total_f1 += f1_stats['f1']
        test_stats['avg'] = total_f1 / len(eval_thresholds)

        display_table(eval_thresholds, test_stats)

        log_stats = {
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'f1@{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
                     
        if(float(test_stats[0.05]) > best_f1_0_05):
            best_f1_0_05 = float(test_stats[0.05])
            with (output_dir / "log_best_f1_0_05.txt").open("w") as f:
                f.write(json.dumps(log_stats) + "\n")
            checkpoint_path = output_dir / 'checkpoint_best_f1_0_05.pth'
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if(test_stats['avg'] > best_f1_avg):
            best_f1_avg = float(test_stats['avg'])
            with (output_dir / "log_best_f1_avg.txt").open("w") as f:
                f.write(json.dumps(log_stats) + "\n")
            checkpoint_path = output_dir / 'checkpoint_best_f1_avg.pth'
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        epoch_list.append(epoch)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args,
                    postprocessors=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    max_norm = args.clip_max_norm

    for locations, features, targets, _, _, coherence_scores \
         in metric_logger.log_every(data_loader, print_freq, header):
        
        features = features.to(device)
        coherence_scores = coherence_scores.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        outputs, mid_scores = model(locations, features, coherence_scores)
        loss_dict = criterion(outputs, mid_scores, targets)
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
        
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, loss_dict

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = general_evaluator()
    val_anno_path = os.path.join(args.annotation_path, 'k400_mr345_val_min_change_duration0.3.pkl')
    with open(val_anno_path,'rb') as f:
        dict_val_ann = pickle.load(f, encoding='lartin1')
    video_pool = list(dict_val_ann.keys())
    video_pool.sort()
    video_dict = {i:video_pool[i] for i in range(len(video_pool))}

    for locations, features, targets, num_frames, base, coherence_scores in metric_logger.log_every(data_loader, 10, header):
        features = features.to(device)
        coherence_scores = coherence_scores.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        outputs, mid_scores = model(locations, features, coherence_scores)

        loss_dict = criterion(outputs, mid_scores, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results = postprocessors['boundaries'](outputs, num_frames, base)


        for target, output in zip(targets, results):
            vid = video_dict[target['video_id'].item()]
            evaluator.update(vid, output)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return evaluator, loss_dict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
 
    # Model parameters

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the encoder")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the decoder")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_cls', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_loc', default=5, type=float,
                        help="L1 dist. coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--loc_loss_coef', default=5, type=float)
    parser.add_argument('--align_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-boundary class")

    # dataset parameters
    parser.add_argument('--dataset', default='kineticsGEBD')
    parser.add_argument('--window_size', default=100, type=int)
    parser.add_argument('--interval', default=3, type=int)
    parser.add_argument('--feature_path', default='/path/to/CONCAT_GEBD_R50_RGB_features', type=str)
    parser.add_argument('--score_path', default='data/', type=str)
    parser.add_argument('--annotation_path', default='data/', type=str)
    parser.add_argument('--thresh', default=0.9, type=float)
    parser.add_argument('--bc_ratio', default=0.8, type=float)
    parser.add_argument('--compress_ratio', default=0.6, type=float)
    
    parser.add_argument('--output_dir', default='outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Temporal Perceiver training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
