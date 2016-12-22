__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"



import numpy as np
from osgeo import gdal
from clipshape import *
from serialize import *
import time
from scipy.ndimage.filters import median_filter
import os

# def sliding_window(image, stepSize, windowSize):
# 	# slide a window across the image
# 	print"inicio..."
# 	#cov = np.empty(image.shape)
# 	median = np.empty(image.shape)
# 	#std = np.empty(image.shape)
# 	for y in xrange(0, image.shape[0], stepSize):
# 		if (y+windowSize[1]) <= image.shape[0]:
# 			for x in xrange(0, image.shape[1], stepSize):
# 				if (x+windowSize[0]) <= image.shape[1]:
# 				# yield the current window
# 					a =  image[y:y + windowSize[1], x:x + windowSize[0]]
# 					# w,v = np.linalg.eig(a)
# 					#cov[y][x] = np.mean(np.cov(a))

# 					median[y][x]= np.rint(np.median(a))
# 					#std[y][x]= np.std(a)
	
# 	return median



# img = gdal.Open("ImgResult/pt611000-4421000Feat4.tiff")
# projection = img.GetProjection()
# geotrans = img.GetGeoTransform()
# image = img.GetRasterBand(1).ReadAsArray()
def moviw(image, imgSavePath, projection,geotrans):
	print "Starting Moving Window"
	start = time.time()
	median  = image
	# for i in range (1):
	# 	# median = medfilt(median,5)
	median = median_filter(median, (3,3), mode = "mirror")

	imgOriginal = gdal.GetDriverByName('GTiff').Create("MovingW"+os.sep+imgSavePath+".tiff", image.shape[0], image.shape[1], 1,gdal.GDT_UInt16,[ 'COMPRESS=LZW' ])
	imgOriginal.GetRasterBand(1).WriteArray(median.T)
	imgOriginal.SetGeoTransform(geotrans)
	imgOriginal.SetProjection(projection)
	print "Image correctly saved in MovingW"+os.sep+"%s" %imgSavePath
	end = time.time()
	print "Time movingwindow:"
	print (end-start)
	return median


# projection, geotrans, imgClass, shpClass = ImageToClassify("/Volumes/LaCie/ortophotos_05022016/pt599000-4413000.tif", False)
# imgResult = read("pickle/images/pt599000-4413000-4")
# imgResult = moviw(imgResult, "pt599000-4413000-4-1iter15x15", projection,geotrans)





