from albumentations import *

EXP_PATH = './experiment_results'
DATA_JSON_PATH = 'train_val_list.json'
PRETRAIN_MODEL_PATH = './pretrained/mobilenetv2_128x128.pth'
SAVE_MODEL_PATH = 'mobilenetv2_v1.pth'

LABELS = [
    '0_0_0', #non-occluded
    '1_0_1', #occluded : bottom
    '1_1_0', #occluded : top
    '1_1_1'  #occluded : all
]

MODEL_CONFIG = {
    'epochs' : 250,
    'lr' : 0.0005,
    'lr_decay' : 'step',
    'momentum' : 0.9,
    'weight_decay' : 1e-4,
    'gamma':0.1,
    'batch_size': 128,
    'worker':0,
    'nc':2,
    'input_size':112,
}

from albumentations.pytorch import ToTensorV2

def get_transform(img_size):
    return [
        Resize(img_size,img_size),
        Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ToTensorV2(),
        ]

def label_dict(n=2):
    if n == 2:
        label_dict = {}
        for k in LABELS:
            if k == '0_0_0':
                label_dict[k] = 0
            else:
                label_dict[k] = 1
    elif n == 4:
        label_dict ={k:v for k,v in zip(LABELS, len(LABELS))}
    return label_dict

def label_decode_dict(n=2):
    if n == 2:
        decode_dict = {}
        for k in LABELS:
            if k == '0_0_0':
                decode_dict[0] = 'non_occluded'
            else:
                decode_dict[1] = 'occluded'
    elif n == 4:
        decode_dict ={v:k for k,v in zip(LABELS, len(LABELS))}
    return decode_dict