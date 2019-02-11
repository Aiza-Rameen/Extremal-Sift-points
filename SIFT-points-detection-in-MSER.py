# Author: Saad Bin Ahmed (copyright)

# This python program helps to understand how extremal regions were extracted from binary image and image mask and later sift features were extracted from the MSER detected regions.
# The paths defined in the following program are specific, programer needs to define it according to file location.
# Download necessary packages e.g., numpy, mtplotlib, opencv etc, in order to have smooth execution of the program. 

# python remove_contours.py

# import the necessary packages
import numpy as np
import cv2, sys, os
#import sift
from PIL import Image
import matplotlib.pyplot as plt

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4


inputFilename = "/input/file/name/sample-images.txt"
#ncFilename= args[1]
print "input filename", inputFilename
#print "data filename", ncFilename
inputFilename=open(inputFilename,"r")
filenames=inputFilename.readlines()
print "filename::::::::::",len(filenames)
pathsave="/input/file/name/Output/"
for  fileline in filenames:
	print " The File path is:::::",fileline[29:]
	filepath=(os.path.splitext(fileline)[0])
	print "FILE PATH",filepath[31:35] # get exact file name.
    	filewithExt=filepath+".JPG"
	thresh = 127
	maxValue = 255
	
	##### load the shapes image, convert it to grayscale, and edge edges in the image #####

	image = cv2.imread(filewithExt)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(gray, 50, 100)
	#cv2.imshow("Original", image)
	th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);

	##### find contours in the image and initialize the mask that will be used to remove the bad contours #####

	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.ones(image.shape[:2], dtype="uint8") * 255

	##### loop over the contours #####

	for c in cnts:
	# if the contour is bad, draw it on the mask
		if is_contour_bad(c):
			cv2.drawContours(mask, [c], -1, 0, -1)


	##### remove the contours from the image and show the resulting images #####

	image = cv2.bitwise_and(image, image, mask=mask)
	cv2.imwrite(pathsave+filepath[31:35]+'_binary.jpg', dst) #Saving
	cv2.imwrite(pathsave+filepath[31:35]+'_mask.jpg', mask) #Saving
	cv2.imwrite(pathsave+filepath[31:35]+'_after.jpg', image) #Saving
	###cv2.imshow("Mask", mask)
	###cv2.imshow("After", image)
	
	image1 = cv2.imread(pathsave+filepath[31:35]+'_binary.jpg')
	mser = cv2.MSER()
	regions = mser.detect(image1, None)
	print "The file path is ..........",filepath[31:35]
	#sys.exit()
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv2.polylines(image1, hulls, 1, (0, 0, 255), 2)
	cv2.imwrite(pathsave+filepath[31:35]+'_mser_bin.jpg',image1)


	image2 = cv2.imread(pathsave+filepath[31:35]+'_mask.jpg')
	mser = cv2.MSER()
	regions = mser.detect(image2, None)
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv2.polylines(image2, hulls, 1, (0, 0, 255), 2)
	cv2.imwrite(pathsave+filepath[31:35]+'_mser_mask.jpg',image2)

	##### Extract SIFT from Binary MSER image  ########################################
	image3 = cv2.imread(pathsave+filepath[31:35]+'_mser_bin.jpg')  # give extremal region detected binary image as an input.
	gftt = cv2.FeatureDetector_create("GFTT")
	kp_list = gftt.detect(image3)
	#print len(kp_list)
	img=cv2.drawKeypoints(image3,kp_list)
	cv2.imwrite(pathsave+filepath[31:35]+'_mser_bin_sift.jpg',img)
	imgplot = plt.imshow(img)
	###plt.show(imgplot)
	

	##### Extract SIFT from MASK MSER image  ########################################
	image4 = cv2.imread(pathsave+filepath[31:35]+'_mser_mask.jpg')  # give extremal region detected image mask as an input.
	gftt = cv2.FeatureDetector_create("GFTT")
	kp_list = gftt.detect(image4)
	#print len(kp_list)
	img1=cv2.drawKeypoints(image4,kp_list)
	cv2.imwrite(pathsave+filepath[31:35]+'_mser_mask_sift.jpg',img1)
	imgplot = plt.imshow(img1)
	###plt.show(imgplot)
