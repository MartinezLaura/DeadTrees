__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"


from clipshape import *
from serialize import *
import os

# Information to provide:
field = "zona" # field in the shapefile where to read the classes
# The following is a subset of the ortophoto, in one file, to which the classes
# are referred
rasterPath  = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Mosaic/Mosaic.tif"
# The following is a shapefile with polygons representing the various classes
shapePath   = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Features/mosaic5.shp"
# Path to the texture layers of the training set
texture_train_Path = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/texture_sample/"
INX = False
pickleclip = "clipfeat-5-text_1"

def init_texture(field, rasterPath, shapePath, INX, texture_train_Path, pickleclip):
    '''
    Create the initialization file (clip)
    '''

    feat, nPixels = ObtainPixelsfromShape(field, \
                                          rasterPath, \
                                          shapePath, \
                                          INX,\
                                          texture_train_Path)
    # INX can be false. If True, uses additional layers.
    Mylist = [feat, nPixels] #pickle wants a list as input

    # Creates the folder if it doesn't exist
    if not os.path.exists("pickle/clip/"):
        os.makedirs("pickle/clip/")
    save("pickle/clip/" + pickleclip, Mylist)

init_texture(field, rasterPath, shapePath, INX, texture_train_Path, pickleclip)
