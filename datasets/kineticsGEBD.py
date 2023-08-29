"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action/blob/main/datasets/thumos14.py)

KineticsGEBD dataset which returns sample list with annotation for training and validation.

"""
from pathlib import Path

import argparse
import torch
import torch.utils.data
import torchvision
import json
import pandas as pd
import os
import numpy as np
import copy
import pickle

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoRecord:
    def __init__(self, vid, num_frames, locations, gt, coherence_scores, fps, args):
        self.id = vid
        self.locations = locations
        self.base = float(locations[0])
        self.window_size = args.window_size
        self.interval = args.interval
        self.rel_locations = [location - self.base for location in locations]
        self.num_frames = num_frames
    
        self.gt = gt
        self.gt_norm = copy.deepcopy(gt)
        self.gt_frames = [(i[0] - self.base) / (self.window_size * self.interval) for i in self.gt_norm]

        self.fps = fps

        range_start = np.max(coherence_scores[:]) - np.min(coherence_scores[:])         
        self.coherence_scores = (coherence_scores[:] - np.min(coherence_scores[:])) / range_start
 


class KineticsGEBD(torch.utils.data.Dataset):
    def __init__(self, feature_folder, score_path, anno_root_path, mode, args):

        self.window_size = args.window_size
        self.feature_folder = feature_folder
        self.video2id_dict = {}
        self.mode = mode
        anno_path= os.path.join(anno_root_path, 'k400_mr345_{}_min_change_duration0.3.pkl'.format(mode))
        with open(anno_path,'rb') as f:
            dict_ann = pickle.load(f, encoding='lartin1')
        
        video_pool = list(dict_ann.keys())
        video_pool.sort()
        self.video2id_dict = {video_pool[i]: i for i in range(len(video_pool))}
        self.sample_list = []

        with open(os.path.join(score_path,'{}_score_sequence.pkl'.format(mode)), 'rb') as f:
            scores = pickle.load(f, encoding='lartin1')
       
        for vid in video_pool:
            vdict = dict_ann[vid]
            num_frames = vdict['num_frames']
            fps = vdict['fps']

            f1_consis = vdict['f1_consis']
            # select the annotation with highest f1 score
            highest = np.argmax(f1_consis)
            annotations = vdict['substages_myframeidx'][highest]
            labels = [1 for _ in annotations]

            coherence_scores_per_vid = scores[vid]['scores']
            frames = scores[vid]['frame_idx']

            num_sampled = len(frames) 
            if num_sampled <= self.window_size:
                locations = np.zeros((self.window_size))
                locations[:num_sampled] = frames
                coherence_scores = np.zeros((self.window_size))
                coherence_scores[:num_sampled] = coherence_scores_per_vid

                gt = [(annotations[idx], labels[idx]) for idx in range(len(annotations))]
                self.sample_list.append(VideoRecord(vid, num_frames, locations, gt, coherence_scores, fps, args))
            else:
                overlap_ratio = 1
                stride = self.window_size // overlap_ratio
                ws_starts = [i * stride for i in range((num_sampled // self.window_size - 1) * overlap_ratio + 1)]
                ws_starts.append(num_sampled - self.window_size)
                
                for ws in ws_starts:
                    locations = frames[ws:ws+self.window_size]
                    coherence_scores = coherence_scores_per_vid[ws:ws+self.window_size]
                    gt = []
                    for idx in range(len(annotations)):
                        anno = annotations[idx]
                        label = labels[idx]
                        if anno >= locations[0] and anno<= locations[-1]:
                            gt.append((anno, label))
                    if self.mode != 'train':
                        self.sample_list.append(VideoRecord(vid, num_frames, locations, gt, coherence_scores, fps, args))
                    elif len(gt) > 0:
                        self.sample_list.append(VideoRecord(vid, num_frames, locations, gt, coherence_scores, fps, args))
                   
    def get_data(self, video:VideoRecord):
        '''
        :param VideoRecord
        :return vid_name, 
        locations : [N, 1], 
        all_props_feature: [N, ft_dim + 2 + pos_dim],
        (gt_start_frame, gt_end_frame): [num_gt, 2]
        '''

        vid = video.id
        num_frames = video.num_frames
        base = video.base
        
        abs_locations = torch.Tensor([location for location in video.locations])
        vid_feature = torch.load(os.path.join(self.feature_folder, vid)).squeeze()
        ft_idxes = [min(torch.div(i, 3, rounding_mode='trunc').int(), vid_feature.shape[0] - 1) for i in abs_locations]
        features = vid_feature[ft_idxes]

        assert features.shape == (self.window_size, 2048), print(features.shape)

        locations = torch.Tensor([location for location in video.rel_locations]) 
        coherence_scores = torch.Tensor(video.coherence_scores)  
        gt_frames = [(c, 0) for c in video.gt_frames]

        targets = {'labels':[], 'boundaries':[], 'video_id':torch.Tensor([self.video2id_dict[vid]])}
        for (center,label) in gt_frames:
            targets['labels'].append(int(label))
            targets['boundaries'].append(center)
        targets['labels'] = torch.LongTensor(targets['labels'])
        targets['boundaries'] = torch.Tensor(targets['boundaries'])
         
        return locations, features, targets, num_frames, base, coherence_scores

    def __getitem__(self, idx):
        return self.get_data(self.sample_list[idx])

    def __len__(self):
        return len(self.sample_list)

def collate_fn(batch):
    target_list, num_frames_list, base_list = [[] for _ in range(3)]
    batch_size = len(batch)
    ft_dim = batch[0][1].shape[-1]
    max_props_num = batch[0][0].shape[0]
    features = torch.zeros(batch_size, max_props_num, ft_dim)
    locations = torch.zeros(batch_size, max_props_num, 1, dtype=torch.double)
    coherence_scores = torch.zeros(batch_size, max_props_num)

    for i, sample in enumerate(batch):
        locations[i, :max_props_num, :] = sample[0].reshape((-1,1))
        features[i, :max_props_num, :] = sample[1]
        target_list.append(sample[2])
        num_frames_list.append(sample[3])
        base_list.append(sample[4])
        coherence_scores[i, :max_props_num] = sample[5]
    num_frames = torch.from_numpy(np.array(num_frames_list))
    base = torch.from_numpy(np.array(base_list))
    
    return locations, features, target_list, num_frames, base, coherence_scores

def build(split, args):
    feature_folder = Path(args.feature_path)
    score_path = Path(args.score_path)
    anno_file = Path(args.annotation_path)

    dataset = KineticsGEBD(feature_folder, score_path, anno_file, split, args)
    return dataset
