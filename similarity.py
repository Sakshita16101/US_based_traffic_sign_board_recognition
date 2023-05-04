# from skimage.metrics import structural_similarity as ssim
import skimage
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
import cv2
from resizeimage import resizeimage


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

SCREEN_WIDTH = 128
SCREEN_HEIGHT = 64
scale_factor = 2

imageA=cv2.imread("/home/sakshita/Music/ALPHA/A.jpg",0)
imageB=cv2.imread("/home/sakshita/Music/ALPHA/B.jpg",0)
imageC=cv2.imread("/home/sakshita/Music/ALPHA/C.jpg",0)
imageD=cv2.imread("/home/sakshita/Music/ALPHA/D.jpg",0)
imageE=cv2.imread("/home/sakshita/Music/ALPHA/E.jpg",0)
imageF=cv2.imread("/home/sakshita/Music/ALPHA/F.jpg",0)
imageG=cv2.imread("/home/sakshita/Music/ALPHA/G.jpg",0)
imageH=cv2.imread("/home/sakshita/Music/ALPHA/H.jpg",0)
imageI=cv2.imread("/home/sakshita/Music/ALPHA/I.jpg",0)
imageJ=cv2.imread("/home/sakshita/Music/ALPHA/J.jpg",0)
imageK=cv2.imread("/home/sakshita/Music/ALPHA/K.jpg",0)
imageL=cv2.imread("/home/sakshita/Music/ALPHA/L.jpg",0)
imageM=cv2.imread("/home/sakshita/Music/ALPHA/M.jpg",0)
imageN=cv2.imread("/home/sakshita/Music/ALPHA/N.jpg",0)
imageO=cv2.imread("/home/sakshita/Music/ALPHA/O.jpg",0)
imageP=cv2.imread("/home/sakshita/Music/ALPHA/P.jpg",0)
imageQ=cv2.imread("/home/sakshita/Music/ALPHA/Q.jpg",0)
imageR=cv2.imread("/home/sakshita/Music/ALPHA/R.jpg",0)
imageS=cv2.imread("/home/sakshita/Music/ALPHA/S.jpg",0)
imageT=cv2.imread("/home/sakshita/Music/ALPHA/T.jpg",0)
imageU=cv2.imread("/home/sakshita/Music/ALPHA/U.jpg",0)
imageV=cv2.imread("/home/sakshita/Music/ALPHA/V.jpg",0)
imageW=cv2.imread("/home/sakshita/Music/ALPHA/W.jpg",0)
imageX=cv2.imread("/home/sakshita/Music/ALPHA/X.jpg",0)
imageY=cv2.imread("/home/sakshita/Music/ALPHA/Y.jpg",0)
imageZ=cv2.imread("/home/sakshita/Music/ALPHA/Z.jpg",0)
Test=cv2.imread("/home/sakshita/Desktop/limit/image2.png",0)
# imageC=cv2.imread("/home/sakshita/Desktop/test1.png",0)
# cv2.imshow("img",imageC)
# cv2.waitKey(0)
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

# scaled_img3 = cv2.resize(imageC[1:50, 0:SCREEN_WIDTH], (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


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

# thres,thres_img3 = cv2.threshold(scaled_img3, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# cv2.imshow("img",thres_imgL)
# cv2.waitKey(0)


# cover1 = resizeimage.resize_cover(scaled_img1, [200, 100], validate=False)
# cover2 = resizeimage.resize_cover(scaled_img2, [200, 100], validate=False)

#
# scaled_img1 = scaled_img1.resize(30,30)
# scaled_img2 = scaled_img2.resize(30,30)

#
# DIM1=(30,30)
# resized = cv2.resize(thres_img1, DIM1, interpolation = cv2.INTER_AREA)
# resized = cv2.resize(thres_img2, DIM1, interpolation = cv2.INTER_AREA)


# dim1=thres_img1.shape
# dim2=thres_img2.shape
# print(dim1)
# print(dim2)

# cv2.imshow("thres_img1",thres_img1)
# cv2.waitKey(0)
#
# cv2.imshow("thres_img2",thres_img2)
# cv2.waitKey(0)




# s = skimage.metrics.structural_similarity(thres_img1, thres_img1)
# print(s)
# difference = cv2.subtract(thres_img1, thres_img2)
# # print(difference)
#
# print (np.linalg.det(difference))



MSEA=mse(thres_imgTest,thres_imgA)
MSEB=mse(thres_imgTest,thres_imgB)
MSEC=mse(thres_imgTest,thres_imgC)
MSED=mse(thres_imgTest,thres_imgD)
MSEE=mse(thres_imgTest,thres_imgE)
MSEF=mse(thres_imgTest,thres_imgF)
MSEG=mse(thres_imgTest,thres_imgG)
MSEH=mse(thres_imgTest,thres_imgH)
MSEI=mse(thres_imgTest,thres_imgI)
MSEJ=mse(thres_imgTest,thres_imgJ)
MSEK=mse(thres_imgTest,thres_imgK)
MSEL=mse(thres_imgTest,thres_imgL)
MSEM=mse(thres_imgTest,thres_imgM)
MSEN=mse(thres_imgTest,thres_imgN)
MSEO=mse(thres_imgTest,thres_imgO)
MSEP=mse(thres_imgTest,thres_imgP)
MSEQ=mse(thres_imgTest,thres_imgQ)
MSER=mse(thres_imgTest,thres_imgR)
MSES=mse(thres_imgTest,thres_imgS)
MSET=mse(thres_imgTest,thres_imgT)
MSEU=mse(thres_imgTest,thres_imgU)
MSEV=mse(thres_imgTest,thres_imgV)
MSEW=mse(thres_imgTest,thres_imgW)
MSEX=mse(thres_imgTest,thres_imgX)
MSEY=mse(thres_imgTest,thres_imgY)
MSEZ=mse(thres_imgTest,thres_imgZ)

l=[]
l.append(MSEA)
l.append(MSEB)
l.append(MSEC)
l.append(MSED)
l.append(MSEE)
l.append(MSEF)
l.append(MSEG)
l.append(MSEH)
l.append(MSEI)
l.append(MSEJ)
l.append(MSEK)
l.append(MSEL)
l.append(MSEM)
l.append(MSEN)
l.append(MSEO)
l.append(MSEP)
l.append(MSEQ)
l.append(MSER)
l.append(MSES)
l.append(MSET)
l.append(MSEU)
l.append(MSEV)
l.append(MSEW)
l.append(MSEX)
l.append(MSEY)
l.append(MSEZ)









# MSEB,MSEC,MSED,MSEE,MSEF,MSEG,MSEH,MSEI,MSEJ,MSEK,MSEL,MSEM,MSEN,MSEO,MSEP,MSEQ,MSER,MSES,MSET,MSEU,MSEV,MSEW,MSEX,MSEY,MSEZ)
print("MSEA",MSEA)
print("MSEB",MSEB)

print("MSEC",MSEC)
print("MSED",MSED)

print("MSEE",MSEE)
print("MSEF",MSEF)

print("MSEG",MSEG)
print("MSEH",MSEH)
print("MSEI",MSEI)
print("MSEJ",MSEJ)


print("MSEK",MSEK)
print("MSEL",MSEL)
print("MSEM",MSEM)
print("MSEN",MSEN)

print("MSEO",MSEO)
print("MSEP",MSEP)
print("MSEQ",MSEQ)
print("MSER",MSER)
print("MSES",MSES)
print("MSET",MSET)

print("MSEU",MSEU)
print("MSEV",MSEV)
print("MSEW",MSEW)
print("MSEX",MSEX)
print("MSEY",MSEY)
print("MSEZ",MSEZ)







l.sort()
print(l[0])

################################################################
#
# MSEA=np.convolve(thres_imgTest,thres_imgA)
# MSEB=np.convolve(thres_imgTest,thres_imgB)
# MSEC=np.convolve(thres_imgTest,thres_imgC)
# MSED=np.convolve(thres_imgTest,thres_imgD)
# MSEE=np.convolve(thres_imgTest,thres_imgE)
# MSEF=np.convolve(thres_imgTest,thres_imgF)
# MSEG=np.convolve(thres_imgTest,thres_imgG)
# MSEH=np.convolve(thres_imgTest,thres_imgH)
# MSEI=np.convolve(thres_imgTest,thres_imgI)
# MSEJ=np.convolve(thres_imgTest,thres_imgJ)
# MSEK=np.convolve(thres_imgTest,thres_imgK)
# MSEL=np.convolve(thres_imgTest,thres_imgL)
# MSEM=np.convolve(thres_imgTest,thres_imgM)
# MSEN=np.convolve(thres_imgTest,thres_imgN)
# MSEO=np.convolve(thres_imgTest,thres_imgO)
# MSEP=np.convolve(thres_imgTest,thres_imgP)
# MSEQ=np.convolve(thres_imgTest,thres_imgQ)
# MSER=np.convolve(thres_imgTest,thres_imgR)
# MSES=np.convolve(thres_imgTest,thres_imgS)
# MSET=np.convolve(thres_imgTest,thres_imgT)
# MSEU=np.convolve(thres_imgTest,thres_imgU)
# MSEV=np.convolve(thres_imgTest,thres_imgV)
# MSEW=np.convolve(thres_imgTest,thres_imgW)
# MSEX=np.convolve(thres_imgTest,thres_imgX)
# MSEY=np.convolve(thres_imgTest,thres_imgY)
# MSEZ=np.convolve(thres_imgTest,thres_imgZ)
# # conv = np.convolve(im1, im2)
# l=[]
# l.append(MSEA)
# l.append(MSEB)
# l.append(MSEC)
# l.append(MSED)
# l.append(MSEE)
# l.append(MSEF)
# l.append(MSEG)
# l.append(MSEH)
# l.append(MSEI)
# l.append(MSEJ)
# l.append(MSEK)
# l.append(MSEL)
# l.append(MSEM)
# l.append(MSEN)
# l.append(MSEO)
# l.append(MSEP)
# l.append(MSEQ)
# l.append(MSER)
# l.append(MSES)
# l.append(MSET)
# l.append(MSEU)
# l.append(MSEV)
# l.append(MSEW)
# l.append(MSEX)
# l.append(MSEY)
# l.append(MSEZ)
#
# print("MSEA",MSEA)
# print("MSEB",MSEB)
#
# print("MSEC",MSEC)
# print("MSED",MSED)
#
# print("MSEE",MSEE)
# print("MSEF",MSEF)
#
# print("MSEG",MSEG)
# print("MSEH",MSEH)
# print("MSEI",MSEI)
# print("MSEJ",MSEJ)
#
#
# print("MSEK",MSEK)
# print("MSEL",MSEL)
# print("MSEM",MSEM)
# print("MSEN",MSEN)
#
# print("MSEO",MSEO)
# print("MSEP",MSEP)
# print("MSEQ",MSEQ)
# print("MSER",MSER)
# print("MSES",MSES)
# print("MSET",MSET)
#
# print("MSEU",MSEU)
# print("MSEV",MSEV)
# print("MSEW",MSEW)
# print("MSEX",MSEX)
# print("MSEY",MSEY)
# print("MSEZ",MSEZ)


w, h = thres_imgTest.shape[::-1]

resA = cv2.matchTemplate(thres_imgA,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resB = cv2.matchTemplate(thres_imgB,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resC = cv2.matchTemplate(thres_imgC,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resD = cv2.matchTemplate(thres_imgD,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resE = cv2.matchTemplate(thres_imgE,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resF = cv2.matchTemplate(thres_imgF,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resG = cv2.matchTemplate(thres_imgG,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resH= cv2.matchTemplate(thres_imgH,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resI = cv2.matchTemplate(thres_imgI,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resJ = cv2.matchTemplate(thres_imgJ,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resK = cv2.matchTemplate(thres_imgK,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resL = cv2.matchTemplate(thres_imgL,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resM = cv2.matchTemplate(thres_imgM,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resN = cv2.matchTemplate(thres_imgN,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resO = cv2.matchTemplate(thres_imgO,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resP = cv2.matchTemplate(thres_imgP,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resQ = cv2.matchTemplate(thres_imgQ,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resR = cv2.matchTemplate(thres_imgR,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resS = cv2.matchTemplate(thres_imgS,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resT = cv2.matchTemplate(thres_imgT,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resU = cv2.matchTemplate(thres_imgU,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resV = cv2.matchTemplate(thres_imgV,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resW = cv2.matchTemplate(thres_imgW,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resX = cv2.matchTemplate(thres_imgX,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resY = cv2.matchTemplate(thres_imgY,thres_imgTest,cv2.TM_CCOEFF_NORMED)
resZ = cv2.matchTemplate(thres_imgZ,thres_imgTest,cv2.TM_CCOEFF_NORMED)

print("patternA",resA)
print("patternB",resB)
print("patternC",resC)
print("patternD",resD)
print("patternE",resE)
print("patternF",resF)
print("patternG",resG)
print("patternH",resH)
print("patternI",resI)
print("patternJ",resJ)
print("patternK",resK)
print("patternL",resL)
print("patternM",resM)
print("patternN",resN)
print("patternO",resO)
print("patternP",resP) 
print("patternQ",resQ)
print("patternR",resR)
print("patternS",resS)
print("patternT",resT)
print("patternU",resU)
print("patternV",resV)
print("patternW",resW)
print("patternX",resX)
print("patternY",resY)
print("patternZ",resZ)




