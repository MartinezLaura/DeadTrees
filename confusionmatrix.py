__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

from osgeo import gdal, ogr
import os
from rtree import index

def Rtree(shppoly):
    '''
    construction of the segmentation tree of the polygons
    '''
    file_idx = index.Rtree('RTREE') #RTREE is the name
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
    '''
    Tells if the point is inside the polygon and gives the count (OK or KO)
    '''

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

    driver = None
    dataSource = None
    pointsLayer = None

    print "Number of points inside polygons:" + len(cm)
    print "List of FIDs of the above polygons:"+ cm



for file in os.listdir("/Volumes/FREECOM/Laura/Pine/shpResult/"):
  if file.endswith(".shp"):
    file = os.path.splitext(file)[0]
    Rtree("/Volumes/LaCie/backup/Laura/Pine/shpResult/" +str(file)+".shp")
Pointinpolygon("VisualInt/visual.shp") #put the path of the points of deadtrees
