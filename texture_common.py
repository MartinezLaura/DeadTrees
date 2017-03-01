__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"

import os
import ctypes
from osgeo import gdal, gdalnumeric, ogr, osr
from osgeo import gdal_array
import numpy as np
import time


#-------------------------------------------------------------------------------
def ListTextureLayers(texturepath):
    for file in os.listdir(texturepath):
        texturelist = []
        if file.endswith(".tif"):
            texturelist.append(file)
    return texturelist


#-------------------------------------------------------------------------------
def LoadTextureLayers(texturepath):

    # Tell GDAL to throw Python exceptions, and register all drivers
    gdal.UseExceptions()
    gdal.AllRegister()

    first1 = texturepath + os.listdir(texturepath)[0]
    print "first1 ", first1

    first = gdal_array.LoadFile(first1, gdal.GA_ReadOnly)

    nrows = int(first.shape[0]); print "nrows ", nrows
    ncols = int(first.shape[1]); print "ncols ", ncols

    images = np.ndarray((nrows, ncols))

    print images.shape
    idx = 0
    img_ds = None

    for file in os.listdir(texturepath):
        idx = idx + 1
        print idx
        img_ds = gdal_array.LoadFile(texturepath + file, gdal.GA_ReadOnly)

        images = np.stack((images, img_ds), axis = 2)
        print images.shape

    print "len(images) ", len(images)


    return images



    # 
    # for file in os.listdir(texturepath):
    #     imgarray = gdalnumeric.LoadFile(file)
    #     shpOriginal = imgarray.shape  # save shape for later
    #     imgOriginal = np.concatenate(imgarray.T)
    #     # img = gdal.Open(file)
    #     imgaux[file] = file.ReadAsArray()
    #     imgaaux[file] = imgaux[file].astype(float)
    #
    # textureArray
    # return textureArray


#-------------------------------------------------------------------------------
def ClipTextureLayers(texturepath, lrY, ulY, lrX, ulX):
    texturelist = ListTextureLayers(texturepath)

    texturearray = LoadTextureLayers(texturepath)

    n = len(texturelist)
    clip = np.empty((n, lrY - ulY, lrX - ulX))
    for i in len(texturelist):
        clip[i] = texturearray[i]

    return clip
