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

def moviw(image, imgSavePath, projection, geotrans):
	print "Starting Moving Window"
	start = time.time()

	median = median_filter(image, (3, 3), mode = "mirror")
	# mirror is a method to fill the last column in a moving window

	imgOriginal = gdal.GetDriverByName('GTiff').Create(imgSavePath + ".tif", \
	                                                   image.shape[0], \
													   image.shape[1], \
													   1, \
													   gdal.GDT_UInt16, \
													   [ 'COMPRESS=LZW' ])

	imgOriginal.GetRasterBand(1).WriteArray(median.T)
	imgOriginal.SetGeoTransform(geotrans)
	imgOriginal.SetProjection(projection)

	print "Image correctly saved in " + "%s.tif" %imgSavePath

	end = time.time()
	print "Time movingwindow:"
	print (end - start)
	return median
