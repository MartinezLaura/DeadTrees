__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"



from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.six import StringIO 
import numpy as np
from sklearn import tree
import time
from sklearn.learning_curve import learning_curve
import sys
from sklearn import cross_validation
from sklearn.cross_validation import KFold
# import matplotlib.pyplot as plt
from osgeo import gdal, gdalnumeric, ogr, osr
from sklearn.cross_validation import StratifiedKFold
from serialize import *
import os
import multiprocessing
import ctypes
#import graficas


shared_array = None

def init(shared_array_base,shape):
  global shared_array
  shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
  #print "shape:"+str(shape)
  shared_array = shared_array.reshape(shape)
  #shared_array = shared_array.astype(np.int32)

def ClassifyMap(a):
  start2 = time.time()
  s,p,m = a

  #print str(p)+" "+str(len(s)),
  shared_array[p:(p+len(s))] = m.predict(s).astype(type(shared_array))
  #shared_array[p:p+len(s)] = np.empty(len(s))
  #start2 = time.time()
  #print str(p)+" "+str(time.time() - start2) + "seg."

class ImageClassifier:


  def __init__(self, Model, Threads, pathpicklemodel):
    self.FromFile = pathpicklemodel
    self.imageClass = None
    self.projection = None
    self.geotrans = None
    self.imgOriginal = None
    self.shpOriginal = None
    self.Threads = Threads
    if not self.FromFile:
      if model == 1:
	self.model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma=1/0.2, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
      elif model == 2:
	self.model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=100, p=1, metric='minkowski', metric_params=None, n_jobs=4)
      else: 
	raise ValueError('NameError: no found this classification method')
    else:
      print "Reading model from pickle"+os.sep+"model"+os.sep+self.FromFile
      self.model = read("pickle"+os.sep+"model"+os.sep+self.FromFile)
      self.model.n_jobs = 1

  def GetProjection(self):
    return self.projection
  
  def GetGeotrans(self):
    return self.geotrans
  def GetShape(self):
    return self.shpOriginal

  #Does the classification methot given:
	  # feat --> dictionari withe the clips of the img with their classes
	  # nPixels, shpTrain --> for the fit and precdiction
	  #imgOriginal, shpOriginal --> for the predict function
	  #Return: image as a matrix with the value of the classification 
  def Train(self,feat,nPixels,imgOriginal, shpClass,classtype):
    print "Training"
    if self.FromFile:
      print "[WW]: training model readed from file."
    #predicted = np.asarray([])
    # X = np.empty((nPixels,shpClass[0]),dtype=int)
    # y = np.empty((nPixels),dtype=np.uint8)

    # #w = np.empty((nPixels),dtype=float)
    # offset=0
    # for i in feat: 
    # 	for j in feat.get(i):
    #  		X[offset:offset+j.shape[0],:] = j
    #  		y[offset:offset+j.shape[0]] = i
    #  	# 	if i == '0':
    #  	# 		w[offset:offset+j.shape[0]] = 2000000
    # 		# else:
    # 		# 	w[offset:offset+j.shape[0]] = 1
    #  		offset+=j.shape[0]



    # # # title = "Learning Curves (KNN)"
    # # # Cross validation with 100 iterations to get smoother mean test and train
    # # # score curves, each time with 20% data randomly selected as a validation set.
    # # # cv = cross_validation.ShuffleSplit(len(y), n_iter=100,
    # # #                                    test_size=0.1, random_state=0)
    # # # estimator = KNeighborsClassifier()
    # # # graficas.plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=8)
    # # # plt.show()


    


    # #Assign the clasification method
    # model = ClasMethod(classtype)
    # print ("Time creating model:")
    # print time.strftime('%l:%M%p %Z on %b %d, %Y')

    # #training of the model
    # model.fit(X, y)
    # print "Time fitting model:"
    # print time.strftime('%l:%M%p %Z on %b %d, %Y')

    # save("pickle/model/"+pathpicklemodel, model)
    
  def ImageToClassify(self,imgClass, Bool):
    print "Reading "+imgClass
    imgarray = gdalnumeric.LoadFile(imgClass)
    self.imgOriginal = np.concatenate(imgarray.T)
    img = gdal.Open(imgClass)
    self.shpOriginal = imgarray.shape

    if Bool ==True:
      imgaux = img.ReadAsArray()
      imgaaux = imgaux.astype(float)
      imgOriginal = gdal.GetDriverByName('MEM').Create('newbands.tif', imgarray.shape[2], imgarray.shape[1], 5,gdal.GDT_UInt16)
      imgOriginal.GetRasterBand(1).WriteArray( ((((imgaaux[3]-imgaaux[0]) / (imgaaux[3]+imgaaux[0]))+1)*127.5).astype(int))
      imgOriginal.GetRasterBand(2).WriteArray(((((imgaaux[1]-imgaaux[0]) / (imgaaux[1]+imgaaux[0]))+1)*127.5).astype(int))
      imgOriginal.GetRasterBand(4).WriteArray((((imgaaux[1]-imgaaux[2]) / (imgaaux[1]+imgaaux[2])+1)*127.5).astype(int))
      imgOriginal.GetRasterBand(4).WriteArray((((imgaaux[0]-imgaaux[2]) / (imgaaux[0]+imgaaux[2])+1)*127.5).astype(int))
      imgOriginal.GetRasterBand(5).WriteArray((((imgaaux[3]-imgaaux[1]) / (imgaaux[3]+imgaaux[1])+1)*127.5).astype(int))
      imgOriginal = imgOriginal.ReadAsArray()
      self.shpOriginal = imgOriginal.shape
      self.imgOriginal = np.concatenate(imgOriginal.T)

    self.projection = img.GetProjection()
    self.geotrans = img.GetGeoTransform()


  def Classify(self):

    #shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    #shared_array = shared_array.reshape(self.imgOriginal.shape[0])

      
    shared_array_base = multiprocessing.Array(ctypes.c_int, self.imgOriginal.shape[0])
    
    pool = multiprocessing.Pool(processes=self.Threads,initializer=init, initargs=(shared_array_base,self.imgOriginal.shape[0],))


    #counter = Value('i', 0)
    #p = Pool()
    print "Starting classification...."
    start = time.time()
    # make predictions
    print self.shpOriginal
    print "Total pixels:"+str(self.imgOriginal.shape[0])
    #predicted = np.empty(self.imgOriginal.shape[0],dtype=int)
    
    size=5000
    #size = 5000
    splits=np.array_split(self.imgOriginal,int(self.imgOriginal.shape[0]/size))
    #print len(splits)
    #print np.arange(0,self.imgOriginal.shape[0],size)
    off=[]
    b=0
    for c in [len(r) for r in splits]:
      off.append(b)
      b+=c
    a = zip(splits,off,[self.model]*len(splits))
    self.imgOriginal = None


    print "Pixels groups:"+str(len(a))+" of size:"+str(size)
    
    pool.map(ClassifyMap, a)

    #for p,s in zip(pos,splited):
      #start2 = time.time()
      #print str(p)+" "+str(len(s)),
      #predicted[p:p+len(s)] = self.model.predict(s)
#start2 = time.time()
      #print str(p)+" "+str(len(s)),
    #predicted = self.model.predict(self.imgOriginal)

    print "Time predicting model:"
    print time.strftime('%l:%M%p %Z on %b %d, %Y')


    #give the original shape to the img in order to save it
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    self.imgClass = shared_array.reshape((self.shpOriginal[2],self.shpOriginal[1]))
    #shared_array = shared_array.reshape((self.shpOriginal[2],self.shpOriginal[1]))
    predicted = None
    shared_array_base = None
    shared_array = None
    end = time.time()
    print "Time classification:"
    print (end-start)


  # #Save the img in the selected folder with the selected Name
  # 	# projection and geotrans-->  in order to save the geographic posicion of the classification image,
  def SaveImg(self,imgSavePath):
    driver = gdal.GetDriverByName('GTiff')
    #tener en cuenta perdida lzw
    print "%s.tiff saved." %imgSavePath
    dataset = driver.Create("%s.tiff" %imgSavePath,self.imgClass.T.shape[1],self.imgClass.T.shape[0],1,gdal.GDT_UInt16,[ 'COMPRESS=LZW' ])

    #Add the geooreferenzzation to the img
    dataset.SetGeoTransform(self.geotrans)  
    dataset.SetProjection(self.projection)
    dataset.GetRasterBand(1).WriteArray(self.imgClass.T)
    dataset.FlushCache()
    
  def GetClassified(self):
    return self.imgClass

  def Metrics(self,arrTrue, arrPredict, labels):
    cm = metrics.confusion_matrix(arrTrue, arrPredict, labels = map(int,labels))
    rep = metrics.classification_report(
	    arrTrue,arrPredict)
    print rep
    print cm
    # return cm,rep

  def CrossValidation(self,X,y,model,labels):
    scores = np.empty(shape=(1,3))	
    averages = np.empty(shape=(1,3))
    #kf = KFold(len(y), n_folds=3)
    skf = StratifiedKFold(y, 10)
    #rs = cross_validation.ShuffleSplit(len(y), n_iter=100,test_size=.2, random_state=0)
    for train, test in skf:
	    inx = 0
	    X_train, X_test = X[train], X[test]
	    y_train, y_test = y[train], y[test]
	    predicted = model.fit(X_train,y_train)
	    y_pred = predicted.predict(X_test)
	    Metrics(y_test, y_pred, labels)
	    score = predicted.score(X_test,y_test)
	    predicted = cross_validation.cross_val_predict(model, X_train,y_train)
	    metricModel = metrics.accuracy_score(y_train, predicted)
	    print score
	    print metricModel
	    scores[inx] = score
	    averages[inx] = metricModel
	    inx += 1
    self.metricModel = np.average(averages)
    self.score = np.average(scores)
    # plot_learning_curve(model, "Learning Curve", X_train, y_train, cv=rs)
    # plt.show()
    #return metricModel,score

  def WriteTxt(self,cm,rep,score,metricModel,imgSavePath):
    archi=open("%s.txt" %imgSavePath,'w')
    archi.write('Confusion Marix \n')
    archi.write('%s \n' %cm)
    archi.write('Other metrics \n')
    archi.write('%s \n' %rep)
    archi.write('Cross Validation \n')
    archi.write('Score\n')
    archi.write('%s \n' %self.score)
    archi.write('Accurancy model\n')
    archi.write('%s \n' %metricModel)
    # archi.write("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    archi.close()
    print "TXT correctly saved in %s" %imgSavePath
