from itertools import groupby
from pycocotools import mask as mutils
from pycocotools.coco import COCO
import sys
print(sys.version)
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import gc

from glob import glob
import matplotlib.pyplot as plt

import torch, torchvision,mmdet
print("torch=",torch.__version__,torch.cuda.is_available())
print("mmdet=",mmdet.__version__)
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmengine.config import Config
base="mask-rcnn_r101_fpn_2x_coco" # 1
# base1="mask-rcnn_x101-64x4d_fpn_2x_coco"
cfg = Config.fromfile(f'./configs/mask_rcnn/{base}.py')
#----------------------------------------------------
width=1024
height=1024

max_epochs=40

batch_size=8
num_classes=1

dataset_type = 'CocoDataset'
classes = ('blood_vessel','glomerulus')
data_root = './datasets/hubmap2023/'
#-----------------------------------------------------
cfg.model.roi_head.bbox_head.num_classes = num_classes
cfg.model.roi_head.mask_head.num_classes = num_classes
#-----------------------------------------------------
cfg.train_pipeline[2]['scale']=(width,height)
cfg.test_pipeline [1]['scale']=(width,height)
#-----------------------------------------------------
cfg.train_dataloader.dataset.type=dataset_type
cfg.train_dataloader.dataset.metainfo=dict(classes=classes)
cfg.train_dataloader.dataset.data_root=data_root
cfg.train_dataloader.dataset.ann_file='./datasets/hubmap2023/annotations_json_c1/coco_annotations_train_all_fold1.json'
cfg.train_dataloader.dataset.data_prefix=dict(img='train/')
cfg.train_dataloader.batch_size=batch_size
cfg.train_dataloader.dataset.pipeline[2]['scale']=(width,height)
#-----------------------------------------------------
cfg.val_dataloader.dataset.type=dataset_type
cfg.val_dataloader.dataset.metainfo=dict(classes=classes)
cfg.val_dataloader.dataset.data_root=data_root
cfg.val_dataloader.dataset.ann_file='./datasets/hubmap2023/annotations_json_c1/coco_annotations_valid_all_fold1.json'
cfg.val_dataloader.dataset.data_prefix=dict(img='train/')
cfg.val_dataloader.dataset.pipeline[1]['scale']=(width,height)
#-----------------------------------------------------
cfg.test_dataloader.dataset.type=dataset_type
cfg.test_dataloader.dataset.metainfo=dict(classes=classes)
cfg.test_dataloader.dataset.data_root=data_root
cfg.test_dataloader.dataset.ann_file='./datasets/hubmap2023/annotations_json_c1/coco_annotations_valid_all_fold1.json'
cfg.test_dataloader.dataset.data_prefix=dict(img='train/')
cfg.test_dataloader.dataset.pipeline[1]['scale']=(width,height)
#------------------------------------------------------
#------------------------------------------------------
cfg.val_evaluator.type='CocoMetric'
cfg.val_evaluator.ann_file='./datasets/hubmap2023/annotations_json_c1/coco_annotations_valid_all_fold1.json'
cfg.val_evaluator.metric=['segm']

cfg.test_evaluator.type='CocoMetric'
cfg.test_evaluator.ann_file='./datasets/hubmap2023/annotations_json_c1/coco_annotations_valid_all_fold1.json'
cfg.test_evaluator.metric=['segm']
#------------------------------------------------------
cfg.train_cfg.max_epochs=max_epochs
cfg.optim_wrapper.type='OptimWrapper'
cfg.optim_wrapper.optimizer=dict(type='AdamW',lr=0.001,weight_decay=0.05,eps=1e-8,betas=(0.9, 0.999))
cfg.default_hooks = dict(logger=dict(type='LoggerHook', interval=200),
                         checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/segm_mAP'))
# cfg.load_from = './pretrainmodel/mrcnn/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
#------------------------------------------------------
# !mkdir -p configs/HuBMAP
config=f'./configs/HubMap/custom_config_{base}_{width}_{height}.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)

print("done!")