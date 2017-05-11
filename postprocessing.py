__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

from osgeo import gdal, gdalnumeric, ogr, osr


print "Inicio"

shape_uri = #route to the COS file
shape_datasource = ogr.Open(shape_uri)
# #not wanted
aqLayer = shape_datasource.ExecuteSQL('SELECT *  FROM NameofCOSfile WHERE COSN5 LIKE "1.%" OR COSN5 LIKE "2.%" OR COSN5 LIKE "3.1.1%"OR COSN5 LIKE "3.2.4.01%"OR COSN5 LIKE "3.2.4.02%" OR COSN5 LIKE "4.%" OR COSN5 LIKE "5.%"')
# #wanted
# #aqLayer = shape_datasource.ExecuteSQL('SELECT *  FROM Export_Output WHERE COSN5 LIKE "3.1.2%" OR COSN5 LIKE "3.1.3%" OR COSN5 LIKE "3.2.1%" OR COSN5 LIKE "3.2.2%" OR COSN5 LIKE "3.2.3%" OR COSN5 LIKE "3.2.4.03%" OR COSN5 LIKE "3.2.4.04%" OR COSN5 LIKE "3.2.4.05%" OR COSN5 LIKE "3.2.4.06%" OR COSN5 LIKE "3.2.4.07%" OR COSN5 LIKE "3.2.4.08%" OR COSN5 LIKE "3.2.4.09%" OR COSN5 LIKE "3.2.4.10%" OR COSN5 LIKE "3.3%" ')



img = gdal.Open('MovingW'+os.sep+'movwindow10.tif')
gt = img.GetGeoTransform()
proj = img.GetProjection()
cols = img.RasterXSize
rows = img.RasterYSize

NoData_value = -9999
Fill_Data = 1
raster_fn = 'test.tif'
pixel_size = 25

# Create the destination data source
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn,cols,rows,1,gdal.GDT_Int32)
target_ds.SetGeoTransform(gt)
target_ds.SetProjection(proj)
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(NoData_value)
band.Fill(Fill_Data)
gdal.RasterizeLayer(target_ds, [1], aqLayer, None, None, [0], ['ALL_TOUCHED=TRUE'])


data1 = img.GetRasterBand(1).ReadAsArray()
data2 = target_ds.GetRasterBand(1).ReadAsArray()
dataout = data1*data2
raster_fn = 'ImgResult'+os.sep+'Clipfeat4refpost.tif'
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn,cols,rows,1,gdal.GDT_Int32)
target_ds.SetGeoTransform(gt)
target_ds.SetProjection(proj)
target_ds.GetRasterBand(1).WriteArray(dataout)
target_ds.FlushCache()
