__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

from osgeo import gdal, ogr
import os
from rtree import index

def Rtree(shppoly):
    file_idx = index.Rtree('RTREE')
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shppoly, 0)
    polylayer = dataSource.GetLayer()
    print polylayer.GetFeatureCount()

    for polys in polylayer:
        env = polys.GetGeometryRef().GetEnvelope()
        file_idx.insert(polys.GetFID(), (env[0],env[2],env[1],env[3]))
    print "listo"
    driver = None
    dataSource = None
    polylayer = None


def Pointinpolygon(shppoint):

    cm = []
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shppoint, 0)
    pointsLayer = dataSource.GetLayer()
    print pointsLayer.GetFeatureCount()

    file_idx = index.Rtree('RTREE')
    print file_idx


    for points in pointsLayer:
        p = points.GetGeometryRef()
        #print p
        a = list(file_idx.intersection((p.GetX(),p.GetY())))
        if not a:
            cm.append(points.GetFID())
    #print len(a)
    driver = None
    dataSource = None
    pointsLayer = None
    print len(cm)
    print cm
    # driver3 = ogr.GetDriverByName("ESRI Shapefile")
    # dataSource3 = driver3.Open(shptile, 0)
    # tilelayer = dataSource3.GetLayer()
    # print tilelayer.GetFeatureCount()

    # outShapefile = "tile.shp"
    # outDriver = ogr.GetDriverByName("ESRI Shapefile")
    # outDataSource = outDriver.CreateDataSource(outShapefile)


    # for tile in tilelayer:
    #     name = tile.GetField("NAME")
    #     print name
    #     outLayer = outDataSource.CreateLayer("tile")
    #     outLayer.CreateFeature(tile)
    #     print type(outLayer)
    #     pointsLayer.Intersection(pointsLayer, outLayer)
    #     print outLayer.GetFeatureCount()
    #     break


    
    # polylayer.Intersection(pointsLayer, outLayer)
    # pointsLayer.ResetReading()
    # for points in pointsLayer:
    #     polylayer.ResetReading()
    #     for polys in polylayer:
    #         if points.GetGeometryRef().Within(polys.GetGeometryRef()):
    #             print "True"
    #             cm["ok"].append(points.GetFID())
    #         # else:
            #     print "False"
            #     cm["ko"].append(points.GetFID())


# Pointinpolygon("shpResult/Mosaic.shp", "VisualInt/visual.shp")

# for file in os.listdir("/Volumes/FREECOM/Laura/Pine/shpResult/"):
#   if file.endswith(".shp"):
#     file = os.path.splitext(file)[0]
#     Rtree("/Volumes/LaCie/backup/Laura/Pine/shpResult/" +str(file)+".shp")
Pointinpolygon("VisualInt/visual.shp")


