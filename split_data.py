import os, json
import shutil

data_json = "train_val_list.json"
target_folder = 'inference'
with open(data_json,'r') as json_file:
    train_val_data = json.load(json_file)

keys = ['train', 'val']
keys2 = ['non_occluded','occluded']

for k in keys:
    target_ = os.path.join(target_folder, k)
    if not os.path.isdir(target_):
        os.mkdir(target_)
    for k2 in keys2:
        target2 = os.path.join(target_, k2)
        if not os.path.isdir(target2):
            os.mkdir(target2)

for k in keys:
    files = train_val_data[k]
    for file in files:
        img_name = os.path.basename(file)
        annot = int(img_name[0])
        dst_file = os.path.join(target_folder, k, keys2[annot], img_name)
        shutil.copy2(file,dst_file)
        a =1
