import os
import cv2
import numpy as np
from PIL import Image
from sklearn import preprocessing
path1="/home/sakshita/Desktop/Noise/horizontal_blur"
path2="/home/sakshita/Desktop/Noise/vertical blur"
path3="/home/sakshita/Desktop/Noise/Noiseless"
# path4=
for file1,file2, file3 in zip(os.listdir(path1), os.listdir(path2), os.listdir(path3)):
	image_path1 = os.path.join(path1, file1)
	image_path2 = os.path.join(path2, file2)
	image_path3 = os.path.join(path3, file3)
	img1 = cv2.imread(image_path1)
	img2 = cv2.imread(image_path2)
	img3 = cv2.imread(image_path3)

	# The assumption here is that if an image contains high variance then there is a wide spread of responses, both
	# 	edge - like and non - edge
	# 	like, representative
	# 	of
	# 	a
	# 	normal, in -focus
	# 	image.But if there is very
	# 	low
	# 	variance, then
	# 	there is a
	# 	tiny
	# 	spread
	# 	of
	# 	responses, indicating
	# 	there
	# 	are
	# 	very
	# 	little
	# 	edges in the
	# 	image.As
	# 	we
	# 	know, the
	# 	more
	# 	an
	# 	image is blurred, the
	# 	less
	# 	edges
	# 	there
	# 	are.

	#BLUR
	variance_of_laplacian1 = cv2.Laplacian(img1, cv2.CV_64F).var()
	variance_of_laplacian2 = cv2.Laplacian(img2, cv2.CV_64F).var()
	variance_of_laplacian3= cv2.Laplacian(img3, cv2.CV_64F).var()
	print("horizontal_blur_image_variance",variance_of_laplacian1)
	print("vertical_blur_image_variance", variance_of_laplacian2)
	print("noiseless_image_variance", variance_of_laplacian3)
	print("   ")

	# CONTARST
	Y = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)[:, :, 0]
	min = np.min(Y)
	max = np.max(Y)
	contrast = (max - min) / (max + min)
	print("horizontal_blur_image_contrast", contrast)

	Y2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)[:, :, 0]
	min2 = np.min(Y2)
	max2 = np.max(Y2)
	contrast2 = (max2 - min2) / (max2 + min2)
	print("vertical_blur_image_contrast", contrast2)

	Y3 = cv2.cvtColor(img3, cv2.COLOR_BGR2YUV)[:, :, 0]
	min3 = np.min(Y3)
	max3 = np.max(Y3)
	contrast3 = (max3 - min3) / (max3 + min3)
	print("noiseless_image_contrast", contrast3)
	print("   ")

	#BRIGHTNESS

	imag = Image.open(image_path1)
	imag = imag.convert('RGB')
	X, Y = 0, 0
	pixelRGB = imag.getpixel((X, Y))
	R, G, B = pixelRGB
	brightness = sum([R, G, B]) / 3
	print("horizontal_blur_image_brightness", brightness)

	imag2 = Image.open(image_path2)
	imag2 = imag2.convert('RGB')
	X, Y = 0, 0
	pixelRGB = imag2.getpixel((X, Y))
	R, G, B = pixelRGB
	brightness2 = sum([R, G, B]) / 3
	print("vertical_blur_image_brightness", brightness2)

	imag3 = Image.open(image_path3)
	imag3 = imag3.convert('RGB')
	X, Y = 0, 0
	pixelRGB = imag3.getpixel((X, Y))
	R, G, B = pixelRGB
	brightness3 = sum([R, G, B]) / 3
	print("noiseless_image_brightness", brightness3)
	print("  ")
	# normalized_blur = preprocessing.scale(variance_of_laplacian3)
	# normalized_brightness = preprocessing.scale(contrast3)
	# normalized_contrast = preprocessing.scale(brightness3)
	#
	# print("nn",normalized_blur)
	# print("nn1",normalized_contrast)
	# print("nn2",normalized_brightness)









