__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"


from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
import pickle
import time

#import matplotlib.pyplot as plt

def ImageToClassify(imgClass, Bool):
  imgarray = gdalnumeric.LoadFile(imgClass)
  imgOriginal = np.concatenate(imgarray.T)
  img = gdal.Open(imgClass)
  shpOriginal = imgarray.shape

  if Bool ==True:
    imgaux = img.ReadAsArray()
    imgaaux = imgaux.astype(float)
    imgOriginal = gdal.GetDriverByName('GTiff').Create('newbands.tif', imgarray.shape[2], imgarray.shape[1], 5,gdal.GDT_UInt16)
    imgOriginal.GetRasterBand(1).WriteArray((((imgaaux[3]-imgaaux[0]) / (imgaaux[3]+imgaaux[0]))+1)*127.5)
    imgOriginal.GetRasterBand(2).WriteArray((((imgaaux[1]-imgaaux[0]) / (imgaaux[1]+imgaaux[0]))+1)*127.5)
    imgOriginal.GetRasterBand(4).WriteArray(((imgaaux[1]-imgaaux[2]) / (imgaaux[1]+imgaaux[2])+1)*127.5)
    imgOriginal.GetRasterBand(4).WriteArray(((imgaaux[0]-imgaaux[2]) / (imgaaux[0]+imgaaux[2])+1)*127.5)
    imgOriginal.GetRasterBand(5).WriteArray((((imgaaux[3]-imgaaux[1]) / (imgaaux[3]+imgaaux[1])+1)*127.5))
    imgOriginal = imgOriginal.ReadAsArray()
    shpOriginal = imgOriginal.shape
    imgOriginal = np.concatenate(imgOriginal.T)

  projection = img.GetProjection()
  geotrans = img.GetGeoTransform()
  return projection, geotrans, imgOriginal, shpOriginal 

#Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate the pixel location of a geospatial coordinate
def world2Pixel(geoMatrix, x, y):
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = int((x - ulX) / xDist)
  line = int((y - ulY) / yDist)
  return (pixel, line)

#Converts a Python Imaging Library array to a gdalnumeric image.
def imageToArray(i):
        '''
        Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
        '''
        a = gdalnumeric.fromstring(i.tobytes(), 'b')
        a.shape = i.im.size[1], i.im.size[0]
        return a

def NewBands(clipFeat, lrY, ulY,lrX, ulX):
  # #new indx
  # clip = np.empty((4,lrY- ulY,lrX- ulX))
  # clip[0] = clipFeat[0].astype(np.uint8)  
  # ndvi = (((clipFeat[3]-clipFeat[0]) / (clipFeat[3]+clipFeat[0]))+1)*127.5
  # clip[1] = ndvi.astype(np.uint8)
  # clip[2] = clipFeat[2].astype(np.uint8) 
  # nirg = ((clipFeat[3]-clipFeat[1]) / (clipFeat[3]+clipFeat[1])+1)*127.5
  # clip[3] = nirg.astype(np.uint8)

  #regular ind
  clip = np.empty((5,lrY- ulY,lrX- ulX)) 
  ndvi = (((clipFeat[3]-clipFeat[0]) / (clipFeat[3]+clipFeat[0]))+1)*127.5
  clip[0] = ndvi.astype(np.uint8)
  gr = (((clipFeat[1]-clipFeat[0]) / (clipFeat[1]+clipFeat[0]))+1)*127.5
  clip[1] = gr.astype(np.uint8)
  bg = ((clipFeat[1]-clipFeat[2]) / (clipFeat[1]+clipFeat[2])+1)*127.5
  clip[2] = bg.astype(np.uint8)
  br = ((clipFeat[0]-clipFeat[2]) / (clipFeat[0]+clipFeat[2])+1)*127.5
  clip[3] = br.astype(np.uint8)
  nirg = ((clipFeat[3]-clipFeat[1]) / (clipFeat[3]+clipFeat[1])+1)*127.5
  clip[4] = nirg.astype(np.uint8)
  
  return clip

def ReadClipArray(lrY, ulY,lrX, ulX,img):
  clip = np.empty((img.RasterCount,lrY- ulY,lrX- ulX))

  #Read only the pixels needed for do the clip
  for band in range(img.RasterCount):
    band += 1
    imgaux = img.GetRasterBand(band).ReadAsArray(ulX,ulY,lrX- ulX,lrY- ulY)
    clip[band-1] = imgaux
  return clip


#Does the clip of the shape
def ObtainPixelsfromShape(field, rasterPath, shapePath, INX):

  # field='zona'
  #img = gdal.Open('Testimg/subset_RGBNIR_affected_trees')
  # shapefile = "Testimg/features.shp"
  # open dataset, also load as a gdal image to get geotransform
  print"Starting clip...."
  start = time.time()

  img = gdal.Open(rasterPath)
  geoTrans = img.GetGeoTransform()
  geoTransaux = img.GetGeoTransform()
  proj = img.GetProjection()
  imgarrayaux = img.ReadAsArray().shape


  #open shapefile
  driver = ogr.GetDriverByName("ESRI Shapefile")
  dataSource = driver.Open(shapePath, 0)
  layer = dataSource.GetLayer()
  clipdic = defaultdict(list)
  count = 0

  #Convert the layer extent to image pixel coordinates, we read only de pixels needed
  for feature in layer:
    minX, maxX, minY, maxY = feature.GetGeometryRef().GetEnvelope()
    geoTrans = img.GetGeoTransform()

    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)
    #print ulX,lrX,ulY,lrY

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    
    if INX == True:
        clip = ReadClipArray(lrY, ulY,lrX, ulX,img)
        clip = NewBands(clip,lrY, ulY,lrX, ulX)

        
    else:
      #print"Indexes = False"
      clip = ReadClipArray(lrY, ulY,lrX, ulX,img)


    # print clip[0]
    #print"Paso 2"
  #Image and shape to return in the same shape as it needs the classificator algorithms
    # imgReturn = np.concatenate(imgaux.T)
    # shapeReturn = imgaux.shape
    #imgReturn = np.concatenate(imgaux.T)
    # shapeReturn = img.ReadAsArray().shape

    # clip = imgarray[:, ulY:lrY, ulX:lrX]
    # print clip[0]


    #EDIT: create pixel offset to pass to new image Projection info
    xoffset =  ulX
    yoffset =  ulY
    #print "Xoffset, Yoffset = ( %d, %d )" % ( xoffset, yoffset )

    # Create a new geomatrix for the image
    geoTrans = list(geoTrans)
    geoTrans[0] = minX
    geoTrans[3] = maxY


    # Map points to pixels for drawing the boundary on a blank 8-bit, black and white, mask image.
    points = []
    pixels = []
    geom = feature.GetGeometryRef()
    pts = geom.GetGeometryRef(0)
    
    [points.append((pts.GetX(p), pts.GetY(p))) for p in range(pts.GetPointCount())]
      
    [pixels.append(world2Pixel(geoTrans, p[0], p[1])) for p in points]
      

    rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(rasterPoly)
    rasterize.polygon(pixels, 0)
    mask = imageToArray(rasterPoly)
    #print mask.shape

    # #SHow the clips of the features
    # plt.imshow(mask)
    # plt.show()

    # Clip the image using the mask into a dict
    temp=gdalnumeric.choose(mask,(clip, np.nan))

    # #SHow the clips of the image
    # plt.imshow(temp[4])
    # plt.show()
    temp = np.concatenate(temp.T)
    temp = temp[~np.isnan(temp[:,0])]


    clipdic[str(feature.GetField(field))].append(temp)
    count += temp.shape[0]
  end = time.time()
  print "Time clipshape:"
  print (end-start)
  return clipdic,count
##########################################################################