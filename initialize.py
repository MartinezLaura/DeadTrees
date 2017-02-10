__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL 3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"


from clipshape import *
from serialize import *
import os


field = 'zona'
rasterPath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Mosaic/Mosaic.tif'
shapePath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Features/Mosaic4-1.shp'
INX = False


def init(field, rasterPath, shapePath, INX):

  pickleclip="clipfeat-4"
  feat, nPixels = ObtainPixelsfromShape(field, \
  rasterPath, \
  shapePath, INX)
  Mylist = [feat,nPixels]
  if not os.path.exists("pickle/clip/"):
    os.mkdir("pickle/clip/")
  save("pickle/clip/"+pickleclip, Mylist)

init(field, rasterPath, shapePath, INX)
