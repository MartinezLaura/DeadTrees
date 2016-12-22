__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"




from PIL import ImageEnhance
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np
import pickle
# img = Image.open('Testimg/orto/Clip1.tif')
# im = ImageEnhance.Contrast(img)
# img = im.enhance(5)
# im = ImageEnhance.Brightness(img)
# img = im.enhance(0.75)
# img.show()
#asi no sale nada mal el tema del contraste. probar mas adelante


# def GetExtent(gt,cols,rows):
#     ''' Return list of corner coordinates from a geotransform

#         @type gt:   C{tuple/list}
#         @param gt: geotransform
#         @type cols:   C{int}
#         @param cols: number of columns in the dataset
#         @type rows:   C{int}
#         @param rows: number of rows in the dataset
#         @rtype:    C{[float,...,float]}
#         @return:   coordinates of each corner
#     '''
#     ext=[]
#     xarr=[0,cols]
#     yarr=[0,rows]

#     for px in xarr:
#         for py in yarr:
#             x=gt[0]+(px*gt[1])+(py*gt[2])
#             y=gt[3]+(px*gt[4])+(py*gt[5])
#             ext.append([x,y])
#         yarr.reverse()
#     return ext

# ext = GetExtent(gt,cols,rows)
# print ext

# def CreatePolygon(ext):
# 	poly = ogr.Geometry(ogr.wkbLinearRing)
# 	poly.AddPoint(ext[0][0], ext[0][1])
# 	poly.AddPoint(ext[1][0], ext[1][1])
# 	poly.AddPoint(ext[2][0], ext[2][1])
# 	poly.AddPoint(ext[3][0], ext[3][1])
# 	poly.AddPoint(ext[0][0], ext[0][1])
# 	# Create polygon
# 	polyresult = ogr.Geometry(ogr.wkbPolygon)
# 	polyresult.AddGeometry(poly)
# 	return polyresult

# polyresult = CreatePolygon(ext)
# print polyresult.ExportToWkt()
# caca = polyresult.Intersection(dataSource)



# (222703, 4)
# (222703,
# [[102 107 107 108]
#  [111 116 117 121]
#  [ 97  99 100 105]
#  ..., 
#  [  3  40  58   4]
#  [  3  44  61   4]
#  [  4  41  59   4]]
# (222703, 4)
# [1 1 1 ..., 4 4 4]
# (222703,)

feat = defaultdict(list)
shpClass =[4, 6940, 6940]
with open('pickle/clipshapes4inx.pickle', 'rb') as handle:
	Mylist = pickle.load(handle)

feat = Mylist[0]
nPixels = Mylist[1]
temp = defaultdict(list).fromkeys(feat)
X = np.empty((nPixels,shpClass[0]),dtype=int)
y = np.empty((nPixels),dtype=np.uint8)


for key, value in feat.iteritems():
	y = np.hstack(key)
	X = np.concatenate(value)
print y,X, X.shape









#import os
#for file in os.listdir("/mydir"):
    #if file.endswith(".txt"):
