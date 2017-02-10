__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL 3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"


from clipshape import *
from serialize import *
import os

# Information to provide:
field = 'zona' # field in the shapefile where to read the classes
# The following is a subset of the ortophoto, in one file, to which the classes
# are referred
rasterPath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Mosaic/Mosaic.tif'
# The following is a shapefile with polygons representing the various classes
shapePath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Features/Mosaic4-1.shp'
INX = False # Classification based on only 4 bands


def init(field, rasterPath, shapePath, INX):
    '''Create the initialization file (clip)
    '''

    pickleclip = "clipfeat-4" # name of the clip
    feat, nPixels = ObtainPixelsfromShape(field, \
    rasterPath, \
    shapePath, INX)
    Mylist = [feat, nPixels] #pickle wants a list as input 

    # Creates the folder if it doesn't exist
    if not os.path.exists("pickle/clip/"):
        os.makedirs("pickle/clip/")
    save("pickle/clip/" + pickleclip, Mylist)

init(field, rasterPath, shapePath, INX)
