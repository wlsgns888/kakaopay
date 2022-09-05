import cv2
from albumentations import *

transform_list = [
    HorizontalFlip(p=1),
    Rotate(30, p=1),
    GaussianBlur(p=1),
    GaussNoise(p=1),
    Blur(p=1),
    RGBShift(p=1),
    ToGray(p=1),
    ColorJitter(p=1),
    RandomBrightness(p=1),
]



img_file = r"D:\Jinhoon\git\kakao\new_train_3\0_0_00_afw3img.jpg"

img = cv2.imread(img_file)
for trans in transform_list:
    transforms = Compose([trans])
    transformed_img = transforms(image=img)['image']
    img_name = trans.__str__().split('(')[0]
    cv2.imwrite(img_name+'.png', transformed_img)
    cv2.imshow(img_name, transformed_img)
    cv2.waitKey(1)
    