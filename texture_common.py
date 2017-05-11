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
    '''This function lists all the texture layers present in the folder
    regardless of their name. This means that for the time being this code is
    suitable for performing the prediction on one tile at time. If in future
    this code has to be repurposed, you need to pass to this function the file
    root name, and here you have to match it with the texture name. 
    '''
    texturelist = []
    for file in os.listdir(str(texturepath)):
        if file.endswith(".tif"):
            texturelist.append(file)
    return texturelist


#-------------------------------------------------------------------------------
# def LoadTextureLayers(texturepath):
#
#     # Tell GDAL to throw Python exceptions, and register all drivers
#     gdal.UseExceptions()
#     gdal.AllRegister()
#
#     first1 = texturepath + os.listdir(texturepath)[0]
#     print "first1 ", first1
#
#     first = gdal_array.LoadFile(first1, gdal.GA_ReadOnly)
#
#     nrows = int(first.shape[0]); print "nrows ", nrows
#     ncols = int(first.shape[1]); print "ncols ", ncols

    # images = np.ndarray((nrows, ncols))
    #
    # print images.shape
    # idx = 0
    # img_ds = None
    #
    # for file in os.listdir(texturepath):
    #     idx = idx + 1
    #     print idx
    #     img_ds = gdal_array.LoadFile(texturepath + file, gdal.GA_ReadOnly)
    #
    #     images = np.stack((images, img_ds), axis = 2)
    #     print images.shape
    #
    # print "len(images) ", len(images)


    # return images



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
# def ClipTextureLayers(texturepath, lrY, ulY, lrX, ulX):
#     texturelist = ListTextureLayers(str(texturepath))
#
#     texturearray = createTextureArray(texturepath)
#
#     n = len(texturelist)
#     clip = np.empty((n, lrY - ulY, lrX - ulX))
#     for i in len(texturelist):
#         clip[i] = texturearray[i]
#
#     return clip

#-------------------------------------------------------------------------------

# texturepath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/texture_sample/"
# orthopath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Mosaic/Mosaic.tif'


def createTextureArray(texturepath, orthopath):
    texturelist = ListTextureLayers(texturepath)

    ortho = gdal.Open(orthopath)
    XOriginal = ortho.RasterXSize
    YOriginal = ortho.RasterYSize
    shpOriginal = [YOriginal, XOriginal]
    print shpOriginal
    projection = ortho.GetProjection()
    geotrans = ortho.GetGeoTransform()

    imgOriginal = gdal.GetDriverByName('MEM').Create('texturesmem.tif', \
                                                             XOriginal, \
                                                             YOriginal, \
                                                             4 + len(texturelist), \
                                                             gdal.GDT_UInt16)

    #add RGBNIR layers
    print "Reading Mosaic"
    imgOriginal.GetRasterBand(1).WriteArray(ortho.GetRasterBand(1).ReadAsArray())
    ortho.FlushCache()
    imgOriginal.GetRasterBand(2).WriteArray(ortho.GetRasterBand(2).ReadAsArray())
    ortho.FlushCache()
    imgOriginal.GetRasterBand(3).WriteArray(ortho.GetRasterBand(3).ReadAsArray())
    ortho.FlushCache()
    imgOriginal.GetRasterBand(4).WriteArray(ortho.GetRasterBand(4).ReadAsArray())
    ortho.FlushCache()
    ortho = None #free the memory

    for i in range(len(texturelist)):

        print "Reading " + str(texturelist[i])
        texture = gdal.Open(texturepath + str(texturelist[i]))
        imgOriginal.GetRasterBand(i + 5).WriteArray((texture.GetRasterBand(1).ReadAsArray()).astype('uint16'))
        texture.FlushCache()
        texture = None

    #imgOriginal = np.concatenate(imgarray.T) #Transpose because of gdal
    imgOriginal.SetGeoTransform(geotrans)
    imgOriginal.SetProjection(projection)


    return imgOriginal, shpOriginal




























































































#
