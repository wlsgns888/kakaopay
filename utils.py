from inspect import getmembers
import torch
from model.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
import os

def get_model(nc, device, use_pretrain=None):
    model = MobileNetV2(num_classes=nc)
    if use_pretrain:
        ckpt = torch.load(use_pretrain, map_location=device)
        model = load_ckpt(model, ckpt)
    return model 

def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            print(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model

def get_img_files(img_dir):
    import glob
    img_exes = ['png', 'jpg', 'jpeg']
    img_files = list()
    for img_exe in img_exes:
        img_files.extend(glob.glob(f'{img_dir}/*.{img_exe}'))
    return img_files

def split_data_set(img_dir):
    import random, os, json
    img_list = get_img_files(img_dir)
    data_distribution = {
        'non_occlued':[],
        'top_occlued':[],
        'bottom_occlued':[],
        'all_occlued':[]
    }
    for img_src in img_list:
        img_file = img_src.split(os.sep)[-1]
        A = img_file[0]
        B = img_file[2]
        C = img_file[4]
        if A == '0' and B =='0' and C =='0':
            key = 'non_occlued'
        elif B =='1' and C =='0':
            key = 'top_occlued'
        elif B =='0' and C =='1':
            key = 'bottom_occlued'
        elif B =='1' and C =='1':
            key = 'all_occlued'

        data_distribution[key].append(img_src)
    split_rate = 0.7
    train_val_dict = {
        'train': [],
        'val': []
    }
    for k,v in data_distribution.items():
        num_data = len(v)
        train_num = int(num_data * split_rate)
        for i, img_path in enumerate(v):
            if i < train_num:
                data_type = 'train'
            else:
                data_type = 'val'
            train_val_dict[data_type].append(img_path)
    with open('train_val_list.json', "w") as json_file:
        json.dump(train_val_dict, json_file, indent=4)


def save_lr_graph(save_folder_path, lr):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(lr, label="Learning rate")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("Learning rate", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "lr.png")
    plt.savefig(save_path)
    plt.close()

def save_loss_graph(save_folder_path, train_loss, valid_loss):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "loss.png")
    plt.savefig(save_path)
    plt.close()

def save_acc_graph(
    save_folder_path,
    train_acc,
    train_f1,
    valid_acc,
    valid_f1,
):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_acc")
    plt.plot(valid_acc, label="valid_acc")
    plt.plot(train_f1, label="train_f1_score")
    plt.plot(valid_f1, label="valid_f1_score")
    plt.xlabel("epoch")
    plt.ylabel("acc, f1-score")
    plt.title("ACC, F1-score", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "acc_f1_score.png")
    plt.savefig(save_path)
    plt.close()

def make_drop_out_img(img_file, mask_img_file):
    import random
    import cv2
    import copy
    try:
        org_img = cv2.imread(img_file)
        mask_img = cv2.imread(mask_img_file)
        img = copy.deepcopy(org_img)
        mask_size = (random.randrange(20,50),random.randrange(20,50))
        
        roi_x, roi_y = 10,30
        roi_w, roi_h = 80, 70
        roi = [roi_x,roi_y,roi_x+roi_w-mask_size[1],roi_y+roi_h-mask_size[0]]

        cv2.imshow('org_img', org_img)
        cv2.imshow('mask_img', mask_img)

        mask_h,mask_w,c = mask_img.shape
        mask_h_ = random.randrange(0,mask_h-mask_size[0])
        mask_w_ = random.randrange(0,mask_w-mask_size[1])
        crop_img_from_mask = mask_img[mask_h_:mask_h_+mask_size[0], mask_w_:mask_w_+mask_size[1],:]
        cv2.imshow('croped_img', crop_img_from_mask)
        
        random_y = random.randrange(roi[1], roi[3])
        random_x = random.randrange(roi[0], roi[2])
        img[random_y:random_y+mask_size[0], random_x:random_x+mask_size[1], :] = crop_img_from_mask
        cv2.imshow('drop_img', img)
        cv2.waitKey(1)
    except:
        img = None
    return img

if __name__ == '__main__':
    #check model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(2,device)
    # x = torch.randn((3, 3, 224, 224)).to(device)
    # output = model(x)
    # print('output size:', output.size())

    import json, glob, random
    import cv2
    with open('train_val_list.json', 'r') as fp:
        data_list = json.load(fp)
    
    cnt = 1000
    mask_img_files = r'\\10.158.10.28\Motional_Database\02.Dataset\05_OpenImage\02_segment_inspected_train_image'
    mask_img_files = glob.glob(mask_img_files+'/*/*.jpg')
    random.shuffle(data_list['train'])
    random.shuffle(mask_img_files)
    for img_file in data_list['train']:
        if cnt > 2000:
            break
        if os.path.basename(img_file)[:5] == '0_0_0':
            
            drop_out_img = make_drop_out_img(img_file,mask_img_files[cnt])
            if drop_out_img is not None:
                cv2.imwrite(f'dropout_img/1_1_1_dropout_img_{cnt}.jpg', drop_out_img)
                cnt+=1
            
            
        PRETRAIN_MODEL_PATH = './experiment_results/drop_out_test2_result/mobilenetv2_v1.pth'
        
    a = 1
    #make_drop_out_img()
