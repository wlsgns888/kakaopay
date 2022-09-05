from albumentations import *
#==================================================================================
#Learning parameter experiments
exp_learning = {
    'lr_0005_step':{
        'epochs':100,
        'lr_decay' : 'step',
        'lr': 0.0005
    },
    'lr_00005_step':{
        'epochs':100,
        'lr_decay' : 'step',
        'lr': 0.00005
    },
    'lr_0005_cos':{
        'epochs':100,
        'lr_decay' : 'cos',
        'lr': 0.0005
    },
    'lr_00005_cos':{
        'epochs':100,
        'lr_decay' : 'cos',
        'lr': 0.00005
    },
    'lr_0005_linear':{
        'epochs':100,
        'lr_decay' : 'linear',
        'lr': 0.0005
    },
    'lr_00005_linear':{
        'epochs':100,
        'lr_decay' : 'linear',
        'lr': 0.00005
    }
}


#==================================================================================
#Augmentation parameter experiments
exp_transform = {
    'v1':{
        'transforms':[
            HorizontalFlip(p=0.5),
            Rotate(30, p=0.5),
        ]
    }
    ,
    'v2':{
        'transforms':[
            HorizontalFlip(p=0.5),
            Rotate(30, p=0.5),
            GaussianBlur(p=0.2),
            GaussNoise(p=0.5),
        ]
    },
    'v3':{
        'transforms':[
            HorizontalFlip(p=0.5),
            Rotate(30, p=0.5),
            GaussianBlur(p=0.2),
            GaussNoise(p=0.5),
            Blur(p=0.2),
            RGBShift(p=0.5),
            ToGray(p=0.3),
        ]
    },
    'v4':{
        'transforms':[
            HorizontalFlip(p=0.5),
            Rotate(30, p=0.5),
            GaussianBlur(p=0.2),
            GaussNoise(p=0.5),
            Blur(p=0.2),
            RGBShift(p=0.5),
            ToGray(p=0.3),
            ColorJitter(p=0.3),
            RandomBrightness(p=0.5),
        ]
    }
}