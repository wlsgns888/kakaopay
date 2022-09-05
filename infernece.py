import torch
from utils import *
from train import set_data_loader
import cv2
import shutil, os, tqdm
from config.config import *

def save_data(dir_, file_name):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    dst_file = os.path.join(dir_, os.path.basename(file_name))
    shutil.copy2(file_name,dst_file)

def run_inference(model_file):
    dst_dir = os.path.dirname(model_file)
    nc = 2
    label_dict = label_decode_dict(nc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(nc, device)
    status = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(status['model'])
    model.to(device)
    
    train_dataset,val_dataset, test_dataset = set_data_loader('train_val_list.json', 1, 0)
    model.eval()
    bsave = False
    
    for i, item in enumerate(test_dataset):
        input = item['img'].to(device)
        output = model(input)
        value, predict = torch.max(output,1)

        org_img =item['org_img']
        file_name = item['file_name']
        gt = int(item['label'])
        predict = int(predict.cpu())

        if bsave:
            save_data('test', file_name)
        if gt != predict:
            error_type = f'pred_{label_dict[predict]}_gt_{label_dict[gt]}'
            dst_path = os.path.join(dst_dir,'FP', error_type)
            save_data(dst_path, file_name)

if __name__ == '__main__':
    #model_file = 'mobilenetv2_v1.pth'
    
    model_file = '.\experiment_results\learning_exp\lr_0005_step\mobilenetv2_v1.pth'
    model_file = '.\experiment_results\drop_out_test2\mobilenetv2_v1.pth'

    run_inference(model_file)