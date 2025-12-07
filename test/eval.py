from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import yaml
import numpy as np
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import datasets
from metrics_st import Evaluator
from networks.models import *

def main(config, i):
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    model_dict = torch.load(model_path)

    # data
    datasets_dict = {"stanford2d3d": datasets.Stanford2D3D,
                     "deep360": datasets.Deep360,
                     "insta23k": datasets.Insta23k,
                     "m3d": datasets.M3D,}

    cf_test = config['test_dataset_' + str(i+1)]
    dataset_val = datasets_dict[cf_test['name']]
    test_dataset = dataset_val(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'], 
                            cf_test['args']['augment_color'],
                            cf_test['args']['augment_flip'],
                            cf_test['args']['augment_rotation'],
                            cf_test['args']['repeat'],
                            is_training=False)
    test_loader = DataLoader(test_dataset, 
                            cf_test['batch_size'], 
                            False,
                            num_workers=cf_test['num_workers'], 
                            pin_memory=True, 
                            drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // cf_test['batch_size']
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    model = make(config['model'])
    if any(key.startswith('module') for key in model_dict.keys()):
        model = nn.DataParallel(model)
        
    model.cuda()
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()

    evaluator = Evaluator(config['median_align'])
    evaluator.reset_eval_metrics()
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["rgb"].cuda()
            outputs = model(equi_inputs)
            
            outputs['pred_mask'] = 1 - outputs['pred_mask']
            outputs['pred_mask'] = (outputs['pred_mask'] > 0.5)
            outputs['pred_depth'][~outputs['pred_mask']] = 1


            pred_depth = outputs['pred_depth'].clone()
            pred_depth =pred_depth.detach().cpu()
            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]

        
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])


    evaluator.print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config, 0)