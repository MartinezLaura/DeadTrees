__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"


from osgeo import gdal, ogr,osr
import numpy as np
import mlh
import time
from movingwindow import *
from osgeo import gdal, gdalnumeric, ogr, osr,gdal_array
import os
# import sys
# # this allows GDAL to throw Python Exceptions
# gdal.UseExceptions()

# #
# #  get raster datasource
# #
# src_ds = gdal.Open('MovingW/pt611000-4421000Feat425.tiff')
# projection = src_ds.GetProjection()
# geotrans = src_ds.GetGeoTransform()
# if src_ds is None:
#     print 'Unable to open %s' % src_filename
#     sys.exit(1)

# try:
#     srcband = src_ds.GetRasterBand(1)
# except RuntimeError, e:
#     # for example, try GetRasterBand(10)
#     print 'Band ( %i ) not found' % band_num
#     print e
#     sys.exit(1)

# print "start"

# srcarray = src_ds.ReadAsArray()
# def poligonize(srcarray, path, shape,projection,geotrans):
def poligonize(shape, file):
    print "Starting Poligonize..."
    start = time.time()
    srcarray[srcarray>1] = 0
    drv = gdal.GetDriverByName('MEM')
    srs = osr.SpatialReference()
    #print type(projection)
    #print projection
    srs.ImportFromWkt(projection)
    src_ds  = drv.Create('',shape[2],shape[1],1,gdal.GDT_UInt16)
    src_ds.SetGeoTransform(geotrans)
    dest_wkt = srs.ExportToWkt()
    src_ds.SetProjection(dest_wkt)
    gdal_array.BandWriteArray(src_ds.GetRasterBand(1),srcarray.T)
    
    
    #mlh.SaveImg(srcarray,'ImgResult'+os.sep+'Poligon'+os.sep+path+'poligonize',projection,geotrans)
    #src_ds = gdal.Open('ImgResult'+os.sep+'Poligon'+os.sep+path+'poligonize.tiff')
    srcband = src_ds.GetRasterBand(1)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( 'shpResult'+os.sep+path + ".shp")


    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shape, 1)
    polylayer=dataSource.GetLayerByIndex(0)

    print "++++++++++++++"
    print polylayer.GetName()
    print polylayer.GetFeatureCount()

    newField = ogr.FieldDefn("caca", ogr.OFTInteger)
    polylayer.CreateField(newField)
    print"llego"

    new_field = ogr.FieldDefn("area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(2) #added line to set precision
    polylayer.CreateField(new_field)
    print"llego"


    for feature in polylayer:
        geom = feature.GetGeometryRef() 
        area = geom.GetArea() 
        feature.SetField("area", area)
        polylayer.SetFeature(feature)

        print geom.GetArea()
        # if (geom.GetArea()  < 0.5) or (geom.GetArea()  > 68):
        #     polylayer.DeleteFeature(feature.GetFID())

    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None )

    print "Shapefile correctly saved in shpResult"+os.sep+"%s" %path
    end = time.time()
    print "Time poligonize:"
    print (end-start)
    polylayer.SyncToDisk()
    dataSource.ExecuteSQL("REPACK "+file)
    print polylayer.GetFeatureCount()
    dataSource = None
# img = gdal.Open('MovingW/pt599000-4413000-4-1iter3x3.tiff')
# imgarray = gdalnumeric.LoadFile('MovingW/pt599000-4413000-4-1iter3x3.tiff')
# projection = img.GetProjection()
# geotrans = img.GetGeoTransform()
# poligonize(imgarray.T, 'pt599000-4413000-4-1iter3x3',projection,geotrans)


for file in os.listdir("/Volumes/FREECOM/Laura/Pine/shpResult/"):
  if file.endswith(".tiff"):
    file = os.path.splitext(file)[0]
    poligonize("/Volumes/FREECOM/Laura/Pine/shpResult/" +str(file)+".shp", file)


