import os
import random
import cv2
import copy
def make_drop_out_img(img_file, mask_img_file):
    org_img = cv2.imread(img_file)
    mask_img = cv2.imread(mask_img_file)
    img = copy.deepcopy(org_img)
    mask_size = (random.randrange(30,50),random.randrange(30,50))
    
    roi = [15,35,15+85-mask_size[1],35+70-mask_size[0]]

    cv2.imshow('org_img', org_img)
    cv2.imshow('mask_img', mask_img)

    mask_h,mask_w,c = mask_img.shape
    mask_h_ = random.randrange(0,mask_h-30)
    mask_w_ = random.randrange(0,mask_w-30)
    crop_img_from_mask = mask_img[mask_h_:mask_h_+mask_size[0], mask_w_:mask_w_+mask_size[1],:]
    cv2.imshow('croped_img', crop_img_from_mask)
    
    random_y = random.randrange(roi[1], roi[3])
    random_x = random.randrange(roi[0], roi[2])
    img[random_y:random_y+mask_size[0], random_x:random_x+mask_size[1], :] = crop_img_from_mask
    cv2.imshow('drop_img', img)
    cv2.waitKey(0)





    


if __name__ == '__main__':
    img = "new_train_3\\0_0_00_afw122img.jpg"
    mask_img = "new_train_3\\1_1_0sunglass_crop_TLMANDYMOOREHEDONYV613.JPG"
    
    make_drop_out_img(img, mask_img)

