from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from keras import backend as K
import os

SCREEN_WIDTH = 128
SCREEN_HEIGHT = 64
scale_factor = 2

imageA=cv2.imread("/home/sakshita/Desktop/ALPHABET/A.png",0)
imageB=cv2.imread("/home/sakshita/Desktop/ALPHABET/B.png",0)
imageC=cv2.imread("/home/sakshita/Desktop/ALPHABET/C.png",0)
imageD=cv2.imread("/home/sakshita/Desktop/ALPHABET/D.png",0)
imageE=cv2.imread("/home/sakshita/Desktop/ALPHABET/E.png",0)
imageF=cv2.imread("/home/sakshita/Desktop/ALPHABET/F.png",0)
imageG=cv2.imread("/home/sakshita/Desktop/ALPHABET/G.png",0)
imageH=cv2.imread("/home/sakshita/Desktop/ALPHABET/H.png",0)
imageI=cv2.imread("/home/sakshita/Desktop/ALPHABET/I.png",0)
imageJ=cv2.imread("/home/sakshita/Desktop/ALPHABET/J.png",0)
imageK=cv2.imread("/home/sakshita/Desktop/ALPHABET/K.png",0)
imageL=cv2.imread("/home/sakshita/Desktop/ALPHABET/L.png",0)
imageM=cv2.imread("/home/sakshita/Desktop/ALPHABET/M.png",0)
imageN=cv2.imread("/home/sakshita/Desktop/ALPHABET/N.png",0)
imageO=cv2.imread("/home/sakshita/Desktop/ALPHABET/O.png",0)
imageP=cv2.imread("/home/sakshita/Desktop/ALPHABET/P.png",0)
imageQ=cv2.imread("/home/sakshita/Desktop/ALPHABET/Q.png",0)
imageR=cv2.imread("/home/sakshita/Desktop/ALPHABET/R.png",0)
imageS=cv2.imread("/home/sakshita/Desktop/ALPHABET/S.png",0)
imageT=cv2.imread("/home/sakshita/Desktop/ALPHABET/T.png",0)
imageU=cv2.imread("/home/sakshita/Desktop/ALPHABET/U.png",0)
imageV=cv2.imread("/home/sakshita/Desktop/ALPHABET/V.png",0)
imageW=cv2.imread("/home/sakshita/Desktop/ALPHABET/W.png",0)
imageX=cv2.imread("/home/sakshita/Desktop/ALPHABET/X.png",0)
imageY=cv2.imread("/home/sakshita/Desktop/ALPHABET/Y.png",0)
imageZ=cv2.imread("/home/sakshita/Desktop/ALPHABET/Z.png",0)
Test=cv2.imread("/home/sakshita/Desktop/limit/image4.png",0)

scaled_imgA = cv2.resize(imageA[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgB = cv2.resize(imageB[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgC = cv2.resize(imageC[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgD = cv2.resize(imageD[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgE = cv2.resize(imageE[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgF = cv2.resize(imageF[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgG = cv2.resize(imageG[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgH = cv2.resize(imageH[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgI = cv2.resize(imageI[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgJ = cv2.resize(imageJ[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgK = cv2.resize(imageK[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgL = cv2.resize(imageL[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgM = cv2.resize(imageM[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgN = cv2.resize(imageN[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgO = cv2.resize(imageO[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgP = cv2.resize(imageP[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgQ = cv2.resize(imageQ[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgR = cv2.resize(imageR[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgS = cv2.resize(imageS[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgT = cv2.resize(imageT[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgU = cv2.resize(imageU[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgV = cv2.resize(imageV[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgW = cv2.resize(imageW[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgX = cv2.resize(imageX[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgY = cv2.resize(imageY[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgZ = cv2.resize(imageZ[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
scaled_imgTest = cv2.resize(Test[1:50, 0:SCREEN_WIDTH], (30,30), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


thres,thres_imgA = cv2.threshold(scaled_imgA, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgB = cv2.threshold(scaled_imgB, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgC = cv2.threshold(scaled_imgC, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgD = cv2.threshold(scaled_imgD, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgE = cv2.threshold(scaled_imgE, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgF = cv2.threshold(scaled_imgF, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgG = cv2.threshold(scaled_imgG, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgH = cv2.threshold(scaled_imgH, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgI = cv2.threshold(scaled_imgI, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgJ = cv2.threshold(scaled_imgJ, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgK = cv2.threshold(scaled_imgK, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgL = cv2.threshold(scaled_imgL, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgM = cv2.threshold(scaled_imgM, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgN = cv2.threshold(scaled_imgN, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgO = cv2.threshold(scaled_imgO, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgP = cv2.threshold(scaled_imgP, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgQ = cv2.threshold(scaled_imgQ, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgR = cv2.threshold(scaled_imgR, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgS = cv2.threshold(scaled_imgS, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgT = cv2.threshold(scaled_imgT, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgU = cv2.threshold(scaled_imgU, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgV = cv2.threshold(scaled_imgV, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgW = cv2.threshold(scaled_imgW, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgX = cv2.threshold(scaled_imgX, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgY = cv2.threshold(scaled_imgY, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thres,thres_imgZ = cv2.threshold(scaled_imgZ, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

thres,thres_imgTest = cv2.threshold(scaled_imgTest, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)


(scoreA, diffA) = compare_ssim(thres_imgTest,thres_imgA, full=True)
diffA = (diffA * 255).astype("uint8")
(scoreB, diffB) = compare_ssim(thres_imgTest,thres_imgB, full=True)
diffB = (diffB * 255).astype("uint8")
(scoreC, diffC) = compare_ssim(thres_imgTest,thres_imgC, full=True)
diffC = (diffC * 255).astype("uint8")
(scoreD, diffD) = compare_ssim(thres_imgTest,thres_imgD, full=True)
diffD = (diffD * 255).astype("uint8")
(scoreE, diffE) = compare_ssim(thres_imgTest,thres_imgE, full=True)
diffE = (diffE * 255).astype("uint8")
(scoreF, diffF) = compare_ssim(thres_imgTest,thres_imgF, full=True)
diffF = (diffF * 255).astype("uint8")
(scoreG, diffG) = compare_ssim(thres_imgTest,thres_imgG, full=True)
diffG = (diffG * 255).astype("uint8")
(scoreH, diffH) = compare_ssim(thres_imgTest,thres_imgH, full=True)
diffH = (diffH * 255).astype("uint8")
(scoreI, diffI) = compare_ssim(thres_imgTest,thres_imgI, full=True)
diffI = (diffI * 255).astype("uint8")
(scoreJ, diffJ) = compare_ssim(thres_imgTest,thres_imgJ, full=True)
diffJ = (diffJ * 255).astype("uint8")
(scoreK, diffK) = compare_ssim(thres_imgTest,thres_imgK, full=True)
diffK = (diffK * 255).astype("uint8")
(scoreL, diffL) = compare_ssim(thres_imgTest,thres_imgL, full=True)
diffL = (diffL * 255).astype("uint8")
(scoreM, diffM) = compare_ssim(thres_imgTest,thres_imgM, full=True)
diffM = (diffM * 255).astype("uint8")
(scoreN, diffN) = compare_ssim(thres_imgTest,thres_imgN, full=True)
diffN = (diffN * 255).astype("uint8")
(scoreO, diffO) = compare_ssim(thres_imgTest,thres_imgO, full=True)
diffO = (diffO * 255).astype("uint8")
(scoreP, diffP) = compare_ssim(thres_imgTest,thres_imgP, full=True)
diffP = (diffP * 255).astype("uint8")
(scoreQ, diffQ) = compare_ssim(thres_imgTest,thres_imgQ, full=True)
diffQ = (diffQ * 255).astype("uint8")
(scoreR, diffR) = compare_ssim(thres_imgTest,thres_imgR, full=True)
diffR = (diffR * 255).astype("uint8")
(scoreS, diffS) = compare_ssim(thres_imgTest,thres_imgS, full=True)
diffS = (diffS * 255).astype("uint8")
(scoreT, diffT) = compare_ssim(thres_imgTest,thres_imgT, full=True)
diffT = (diffT * 255).astype("uint8")
(scoreU, diffU) = compare_ssim(thres_imgTest,thres_imgU, full=True)
diffU = (diffU * 255).astype("uint8")
(scoreV, diffV) = compare_ssim(thres_imgTest,thres_imgV, full=True)
diffV = (diffV * 255).astype("uint8")
(scoreW, diffW) = compare_ssim(thres_imgTest,thres_imgW, full=True)
diffW = (diffW * 255).astype("uint8")
(scoreX, diffX) = compare_ssim(thres_imgTest,thres_imgX, full=True)
diffX = (diffX * 255).astype("uint8")
(scoreY, diffY) = compare_ssim(thres_imgTest,thres_imgY, full=True)
diffY = (diffY * 255).astype("uint8")
(scoreZ, diffZ) = compare_ssim(thres_imgTest,thres_imgZ, full=True)
diffZ = (diffW * 255).astype("uint8")


# print(diffT)


print("scoreA",scoreA)
print("scoreB",scoreB)
print("scoreC",scoreC)
print("scoreD",scoreD)

print("scoreE",scoreE)
print("scoreF",scoreF)
print("scoreG",scoreG)
print("scoreH",scoreH)

print("scoreI",scoreI)
print("scoreJ",scoreJ)
print("scoreK",scoreK)
print("scoreL",scoreL)
print("scoreM",scoreM)
print("scoreN",scoreN)

print("scoreO",scoreO)
print("scoreP",scoreP)
print("scoreQ",scoreQ)
print("scoreR",scoreR)
print("scoreS",scoreS)
print("scoreT",scoreT)
print("scoreU",scoreU)
print("scoreV",scoreV)
print("scoreW",scoreW)
print("scoreX",scoreX)
print("scoreY",scoreY)
print("scoreZ",scoreZ)


########################dice score
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice



#####################################JACARD
img_test=np.array(thres_imgTest).ravel()
img_A=np.array(thres_imgT).ravel()
iouA = jaccard_similarity_score(img_test, img_A)
print(iouA)
img_A=np.array(thres_imgT).ravel()
iouA = jaccard_similarity_score(img_test, img_A)
print(iouA)

img_B=np.array(thres_imgB).ravel()
iouB = jaccard_similarity_score(img_test, img_B)
print(iouB)

img_C=np.array(thres_imgC).ravel()
iouC = jaccard_similarity_score(img_test, img_C)
print(iouC)

img_D=np.array(thres_imgD).ravel()
iouD = jaccard_similarity_score(img_test, img_D)
print(iouD)

img_E=np.array(thres_imgE).ravel()
iouE = jaccard_similarity_score(img_test, img_E)
print(iouE)

img_F=np.array(thres_imgF).ravel()
iouF = jaccard_similarity_score(img_test, img_F)
print(iouF)

img_G=np.array(thres_imgG).ravel()
iouG = jaccard_similarity_score(img_test, img_G)
print(iouG)

img_H=np.array(thres_imgH).ravel()
iouH = jaccard_similarity_score(img_test, img_H)
print(iouH)

img_I=np.array(thres_imgI).ravel()
iouI = jaccard_similarity_score(img_test, img_I)
print(iouI)

img_J=np.array(thres_imgJ).ravel()
iouJ = jaccard_similarity_score(img_test, img_J)
print(iouJ)

img_K=np.array(thres_imgK).ravel()
iouK = jaccard_similarity_score(img_test, img_K)
print(iouK)

img_L=np.array(thres_imgL).ravel()
iouL = jaccard_similarity_score(img_test, img_L)
print(iouL)

img_M=np.array(thres_imgM).ravel()
iouM = jaccard_similarity_score(img_test, img_M)
print(iouM)

img_N=np.array(thres_imgN).ravel()
iouN = jaccard_similarity_score(img_test, img_N)
print(iouN)

img_O=np.array(thres_imgO).ravel()
iouO = jaccard_similarity_score(img_test, img_O)
print(iouO)

img_P=np.array(thres_imgP).ravel()
iouP = jaccard_similarity_score(img_test, img_P)
print(iouP)


img_Q=np.array(thres_imgQ).ravel()
iouQ = jaccard_similarity_score(img_test, img_Q)
print(iouQ)

img_R=np.array(thres_imgR).ravel()
iouR = jaccard_similarity_score(img_test, img_R)
print(iouR)

img_S=np.array(thres_imgS).ravel()
iouS = jaccard_similarity_score(img_test, img_S)
print(iouS)

img_T=np.array(thres_imgT).ravel()
iouT = jaccard_similarity_score(img_test, img_T)
print(iouT)

img_U=np.array(thres_imgU).ravel()
iouU = jaccard_similarity_score(img_test, img_U)
print(iouU)

img_V=np.array(thres_imgV).ravel()
iouV = jaccard_similarity_score(img_test, img_V)
print(iouV)
img_W=np.array(thres_imgW).ravel()
iouW = jaccard_similarity_score(img_test, img_W)
print(iouW)

img_X=np.array(thres_imgX).ravel()
iouX = jaccard_similarity_score(img_test, img_X)
print(iouX)

img_Y=np.array(thres_imgY).ravel()
iouY = jaccard_similarity_score(img_test, img_Y)
print(iouY)

img_Z=np.array(thres_imgZ).ravel()
iouZ = jaccard_similarity_score(img_test, img_Z)
print(iouZ)


def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
path="/home/sakshita/Desktop/ALPHABET"
for files in os.listdir(path):
    path2=path+"/"+files
    print(files)
    image = cv2.imread(path2, 0)
    scaled_img = cv2.resize(imageA[1:50, 0:SCREEN_WIDTH], (30, 30), fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_CUBIC)
    thres, thres_img = cv2.threshold(scaled_imgA, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    MSE=mse()
    # dice=dice_coef(thres_img,thres_imgTest)
    # print(dice)






