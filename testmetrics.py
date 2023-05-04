from skimage.measure import compare_ssim
import argparse
import imutils
import cv2# from keras import backend as K

import numpy as np
from sklearn.metrics import jaccard_similarity_score
# from keras import backend as K
import os
from math import log10, sqrt

SCREEN_WIDTH = 128
SCREEN_HEIGHT = 64
scale_factor = 3
path="/home/sakshita/Music/ALPHA"

Test=cv2.imread("/home/sakshita/Desktop/limit/4.png",0)
scaled_imgTest = cv2.resize(Test[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
thres,thres_imgTest = cv2.threshold(scaled_imgTest, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)



def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

Mse=[]

Ssi=[]

Iou=[]

Dice=[]

Psnr=[]
img_test=np.array(thres_imgTest).ravel()

for files in os.listdir(path):
    path2=path+"/"+files
    print(files)
    image = cv2.imread(path2, 0)

    scaled_img = cv2.resize(image[1:50, 0:SCREEN_WIDTH], (30, 30), fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_CUBIC)
    thres, thres_img = cv2.threshold(scaled_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("img", thres_img)
    # cv2.waitKey(0)
    MSE=mse(thres_imgTest,thres_img)
    print("Mean squared error",MSE)
    Mse.append(MSE)

    (score, diff) = compare_ssim(thres_imgTest, thres_img, full=True)
    print("Structural Similarity Index",score)
    Ssi.append(score)

    img = np.array(thres_img).ravel()
    iou = jaccard_similarity_score(img_test, img)
    print("Jaccard Similarity Index",iou)
    Iou.append(iou)

    value = PSNR(thres_img, thres_imgTest)
    print(f"PSNR value is ",value)
    Psnr.append(value)






print("values for character")
cv2.imshow("img",thres_imgTest)
cv2.waitKey(0)
# Mse.sort()
print("MSE",min(Mse))
# Ssi.sort()
print("SSI",max(Ssi))
# Iou.sort()
print("JACCARD INDEX",max(Iou))
print("PSNR",min(Psnr))