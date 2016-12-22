__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"




from clipshape import *
from mlh import *
from serialize import *
from movingwindow import *
from poligonize import *
#from multiprocessing import Pool



feat = defaultdict(list)

def main(nameresult):
  print nameresult
  if not os.path.isfile('shpResult'+os.sep+nameresult + ".shp"):
    pathpickleclip="clipfeat-4"
    # feat, nPixels = ObtainPixelsfromShape('zona', 'Testimg/Mosaic/Mosaic.tif', 'Testimg/Features/Mosaic4-1.shp', False)
    # Mylist = [feat,nPixels]
    # save("pickle/clip/"+pathpickleclip, Mylist)
#	  path = "/media/sf_artesto/temporal/test/"
    path = "/media/sf_artesto/temporal/ortophotos_05022016/"
    #path = "E:\\artesto\\temporal\\ortophotos_05022016\\"
    pathimg=path+nameresult
    Mylist = read("pickle"+os.sep+"clip"+os.sep+pathpickleclip)
    feat = Mylist[0]
    nPixels = Mylist[1] 

    projection, geotrans, imgClass, shpClass = ImageToClassify(pathimg+".tif", False)

    imgResult = ClassificationTool(feat,nPixels,imgClass, shpClass, 2, pathpicklemodel)

    SaveImg(imgResult,"ImgResult"+os.sep+nameresult,projection,geotrans)
    save("pickle"+os.sep+"images"+os.sep+nameresult, imgResult)
    imgResult = read("pickle"+os.sep+"images"+os.sep+nameresult)

    imgResult = moviw(imgResult, nameresult, projection,geotrans)
    poligonize(imgResult, nameresult,projection,geotrans)



# shapePath = "Testimg/Features/Tilestouse.shp"
# driver = ogr.GetDriverByName("ESRI Shapefile")
# dataSource = driver.Open(shapePath, 0)
# layer = dataSource.GetLayer()
# a = []
# for i in layer:
# 	a.append(i.GetField("NAME"))
#path = "/media/sf_artesto/temporal/test/"
path = "/media/sf_artesto/temporal/ortophotos_05022016/"
#path = "E:\\artesto\\temporal\\ortophotos_05022016\\"
#p = Pool(processes=2)
todo=[]
count=0
pathpicklemodel="modelfeat-4"
Classifier = ImageClassifier(Model = 2, Threads = 4, pathpicklemodel = pathpicklemodel)
for file in os.listdir(path):
  if file.endswith(".tif"):
    file = os.path.splitext(file)[0]
    if not os.path.isfile('shpResult'+os.sep+file + ".shp"):
    #todo.append(file)
      #print file
      Classifier.ImageToClassify(path + os.sep +str(file)+".tif", False)
      Classifier.Classify()
      Classifier.SaveImg("ImgResult"+os.sep+str(file))
      #imgResult = read("pickle"+os.sep+"images"+os.sep+nameresult)
      imgResult = moviw(Classifier.GetClassified(), str(file)+"_MW", Classifier.GetProjection(), Classifier.GetGeotrans())
      poligonize(imgResult, str(file), Classifier.GetShape(),Classifier.GetProjection(), Classifier.GetGeotrans())
      count+=1
      if count > 4:
	break
    # if (file in a):
    #file="pt604000-4395000"
    #print file
#p.map(main,todo)
#main(todo[0])

# main("/Volumes/LaCie/ortophotos_05022016/pt599000-4413000", "pt599000-4413000-4-1", "clipfeat-4-1", "modelfeat-4-1")
