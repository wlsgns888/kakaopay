import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
#File Name : A_B_Cxxxxxxxxx.확장자
#- A(얼굴 전체 중 occluded 여부): 1(Occluded), 0(Non-occluded)
#- B(얼굴 상관(눈 주위) occluded 여부): 1(Occluded), 0(Non-occluded)
#- C(얼굴 하관 occluded 여부): 1(Occluded), 0(Non-occluded)

data_dir = 'new_train_3'
out_dir = 'EDA'
img_exes = ['png', 'jpg', 'jpeg']
img_files = list()
for img_exe in img_exes:
    img_files.extend(glob.glob(f'{data_dir}/*.{img_exe}'))
img_files = [x.split(os.sep)[-1] for x in img_files]

total_img = len(img_files)
data_distribution = {
    'overall': {
        '0':0, #'non_occluded'
        '1':0  #'occluded'
    },
    'top': {
        '0':0, #'non_occluded'
        '1':0  #'occluded'
    },
    'bottom': {
        '0':0, #'non_occluded'
        '1':0  #'occluded'
    },
    'total':{
        'non_occlued':0,
        'top_occlued':0,
        'bottom_occlued':0,
        'all_occlued':0,
    }
}
for img_file in img_files:
    A = img_file[0]
    B = img_file[2]
    C = img_file[4]
    data_distribution['overall'][A] +=1
    data_distribution['top'][B] +=1
    data_distribution['bottom'][C] +=1

    if A == '0' and B =='0' and C =='0':
        data_distribution['total']['non_occlued'] += 1
    elif B =='1' and C =='0':
        data_distribution['total']['top_occlued'] += 1
    elif B =='0' and C =='1':
        data_distribution['total']['bottom_occlued'] += 1
    elif B =='1' and C =='1':
        data_distribution['total']['all_occlued'] += 1
    else:
        print('Error : unknown type', img_file)

for k,v in data_distribution.items():
    plt.figure(figsize = (6,5))
    plt.title(k)
    plt.bar(list(range(len(v.values()))), list(v.values()))
    plt.xticks(list(range(len(v.values()))),list(v.keys()))
    plt.ylabel('count')
    plt.savefig(f'{out_dir}/{k}.png')
    plt.close()

print(data_distribution)
    
