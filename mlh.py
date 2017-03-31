__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"


#
# from sklearn import datasets
# from sklearn import metrics
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.externals.six import StringIO
import numpy as np
from sklearn import tree
import time
from sklearn.learning_curve import learning_curve
import sys
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from osgeo import gdal, gdalnumeric, ogr, osr
from sklearn.cross_validation import StratifiedKFold
from serialize import *
import os
import multiprocessing
import ctypes
from texture_common import *



shared_array = None

def init(shared_array_base, shape):
    global shared_array
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    #shared_array = shared_array.astype(np.int32)

def ClassifyMap(a):
    start2 = time.time()
    s, p, m = a
    # s is the split
    # p is the position
    # m is the model


    #print str(p)+" "+str(len(s)),
    shared_array[p : (p + len(s))] = m.predict(s).astype(type(shared_array))
    # print "shared_array[p : (p + len(s))]", shared_array[p : (p + len(s))]
    #shared_array[p:p+len(s)] = np.empty(len(s))
    #start2 = time.time()
    #print str(p)+" "+str(time.time() - start2) + "seg."

class ImageClassifier:


    def __init__(self, modeltype, Threads, picklemodel, model):
        self.FromFile = picklemodel
        self.imageClass = None
        self.projection = None
        self.geotrans = None
        self.imgOriginal = None
        self.shpOriginal = None
        self.Threads = Threads
        self.model = model

        if self.model == None:

            if modeltype == 1:
                self.model = SVC(C = 1.0, \
                       cache_size = 200, \
                       class_weight = None, \
                       coef0 = 0.0, \
                       decision_function_shape = None, \
                       degree = 3, \
                       gamma = 1 / 0.2, \
                       kernel = 'rbf', \
                       max_iter = -1, \
                       probability = False, \
                       random_state = None, \
                       shrinking = True, \
                       tol = 0.001, \
                       verbose = False)

            elif modeltype == 2:
                self.model = KNeighborsClassifier(n_neighbors = 5, \
                                            weights = 'distance', \
                                            algorithm = 'ball_tree', \
                                            leaf_size = 100, \
                                            p = 1, \
                                            metric = 'minkowski', \
                                            metric_params = None, \
                                            n_jobs = -1)

            elif modeltype == 3:
                self.model = RadiusNeighborsClassifier(radius = 50.0, \
                                                 weights = 'distance', \
                                                 algorithm = 'ball_tree', \
                                                 leaf_size = 700, \
                                                 p = 2, \
                                                 metric = 'minkowski', \
                                                 outlier_label = None, \
                                                 metric_params = None)


            else:
                raise ValueError('NameError: classification method not found')


    # else:
    #   print "Reading model from pickle: " + self.FromFile
    #   self.model = read("pickle" + os.sep + "model" + os.sep + self.FromFile)
    #   self.model.n_jobs = 1

    def GetProjection(self):
        return self.projection

    def GetGeotrans(self):
        return self.geotrans

    def GetShape(self):
        return self.shpOriginal

  #Does the classification methot given:
	  # feat --> dictionary with the clips of the img with their classes
	  # nPixels, shpTrain --> for the fit and prediction
	  # imgOriginal, shpOriginal --> for the predict function
	  # Return: image as a matrix with the value of the classification

    def Train(self, feat, nPixels, layer, MyName):
        start = time.time()
        print "Training"
        predicted = np.asarray([])
        X = np.empty((nPixels, layer), dtype = int)
        y = np.empty((nPixels), dtype = np.uint8)

        # Useful if you use the random forest classifier
        # #w = np.empty((nPixels),dtype=float) #for random forest classifier
        offset = 0
        for i in feat:
            for j in feat.get(i):
                X[offset : offset + j.shape[0], :] = j
                y[offset : offset + j.shape[0]] = i
                offset+=j.shape[0]
                # print "j.shape[0]", j.shape[0]
                # print "offset", offset
        # print "X.shape" , X.shape
        # To assign the w
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
        # test_size is the percentage of the traning set
        # # # estimator = KNeighborsClassifier()
        # # # graficas.plot_learning_curve(estimator, \
                                        #    title, \
                                        #    X, \
                                        #    y, \
                                        #    ylim = (0.7, 1.01), \
                                        #    cv = cv, \
                                        #    n_jobs = 8)
        # # # plt.show()


        # #Assign the clasification method
        #model = ClasMethod(classtype)
        print ("Time creating model:")
        print time.strftime('%l:%M%p %Z on %b %d, %Y')

        #training of the model
        self.model.fit(X, y)
        print "Time fitting model:"
        print time.strftime('%l:%M%p %Z on %b %d, %Y')

        # creates the model directory if it doesn't exist
        if not os.path.exists("pickle/model/"):
            os.makedirs("pickle/model/")


        save("pickle/model/" + str(MyName), self.model)


    def ImageToClassify(self, imgClass, Bool, *args):
        '''Prepares the images into a matrix, that has each layer as a column,
        and the rows are the pixels.
        imgClass is the image to classify (orthophoto)
        Bool = False takes R G B NIR
        Bool = True takes all the other layers, for example indexes, texture etc
        '''
        print "Reading " + imgClass


        if Bool == True:
            #   imgaux = img.ReadAsArray()
            #   imgaaux = imgaux.astype(float)

            # imgaux[0] = Red
            # imgaux[1] = green
            # imgaux[2] = blue
            # imgaux[3] = nir
            # if you add new layers, add them here


            texturepath = args[0]
            # print texturepath
            img, self.shpOriginal = createTextureArray(texturepath, imgClass)
            # print "type(img) ", type(img)
            # print "self.shpOriginal ", self.shpOriginal

        else:
            img = gdal.Open(imgClass)
            # print "type(img) else", type(img)
            XOriginal = img.RasterXSize
            YOriginal = img.RasterYSize
            self.shpOriginal = [YOriginal, XOriginal]

        imgarray = img.ReadAsArray()
        # print "type(imgarray)", type(imgarray)
        self.projection = img.GetProjection()
        self.geotrans = img.GetGeoTransform()
        self.imgOriginal = np.concatenate(imgarray.T) #Transpose because of gdal
        imgarray = None

        # self.projection = self.imgOriginal.GetProjection()
        # self.geotrans = self.imgOriginal.GetGeoTransform()

        # print "self.imgOriginal", self.imgOriginal


    def Classify(self):

        #shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        #shared_array = shared_array.reshape(self.imgOriginal.shape[0])

        shared_array_base = multiprocessing.Array(ctypes.c_int, \
                                                  self.imgOriginal.shape[0])

        pool = multiprocessing.Pool(processes = self.Threads, \
                                    initializer = init, \
                                    initargs = (shared_array_base, \
                                    self.imgOriginal.shape[0]))



        print "Starting classification...."
        start = time.time()
        # make predictions
        # print self.shpOriginal
        print "Total pixels:" + str(self.imgOriginal.shape[0])

        # predicted = np.empty(self.imgOriginal.shape[0], dtype = int)

        size = 5000
        #size of the matrix
        # off is the positions of the pixels
        # splits are the cuts
        splits = np.array_split(self.imgOriginal, \
                            int(self.imgOriginal.shape[0] / size))

        # print "len(splits): ", len(splits)
        # print np.arange(0, self.imgOriginal.shape[0], size)
        off = []
        b = 0
        # here we save the positions
        for c in [len(r) for r in splits]:
            off.append(b)
            b += c
        a = zip(splits, off, [self.model] * len(splits))
        self.imgOriginal = None

        # print "off" , off


        print "Pixels groups: " + str(len(a)) + " of size: " + str(size)
        # print "a", a

        pool.map(ClassifyMap, a)

        print "Time predicting model: "
        print time.strftime('%l:%M%p %Z on %b %d, %Y')


        #give the original shape to the img in order to save it
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.imgClass = shared_array.reshape((self.shpOriginal[1], \
                                              self.shpOriginal[0]))

        predicted = None
        shared_array_base = None
        shared_array = None
        end = time.time()
        print "Time classification:"
        print (end - start)


    # Save the img in the selected folder with the selected Name
    # projection and geotrans-->  in order to save the geographic position of
    # the classification image

    def SaveImg(self, imgSavePath):
        driver = gdal.GetDriverByName('GTiff')
        #tener en cuenta perdida lzw
        print "%s.tif saved." %imgSavePath
        dataset = driver.Create("%s.tif" %imgSavePath, \
                             self.imgClass.T.shape[1], \
                             self.imgClass.T.shape[0], \
                             1, \
                             gdal.GDT_UInt16, \
                             [ 'COMPRESS=LZW' ])

        #Add the geooreferenzzation to the img
        dataset.SetGeoTransform(self.geotrans)
        dataset.SetProjection(self.projection)
        dataset.GetRasterBand(1).WriteArray(self.imgClass.T)
        dataset.FlushCache()

    def GetClassified(self):
        return self.imgClass

    def Metrics(self,arrTrue, arrPredict, labels):
        cm = metrics.confusion_matrix(arrTrue, \
                                      arrPredict, \
                                      labels = map(int,labels))
        rep = metrics.classification_report(
                                      arrTrue, \
                                      arrPredict)
        # print rep
        # print cm
        # return cm,rep

    def CrossValidation(self, X, y, model, labels):
        scores = np.empty(shape = (1, 3))
        averages = np.empty(shape = (1, 3))
        #kf = KFold(len(y), n_folds=3)
        skf = StratifiedKFold(y, 10)
        #rs = cross_validation.ShuffleSplit(len(y), n_iter=100, test_size=.2, random_state=0)

        for train, test in skf:
            inx = 0
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            predicted = model.fit(X_train, y_train)
            y_pred = predicted.predict(X_test)
            Metrics(y_test, y_pred, labels)
            score = predicted.score(X_test, y_test)
            predicted = cross_validation.cross_val_predict(model, X_train, y_train)
            metricModel = metrics.accuracy_score(y_train, predicted)
            # print score
            # print metricModel
            scores[inx] = score
            averages[inx] = metricModel
            inx += 1
        self.metricModel = np.average(averages)
        self.score = np.average(scores)
        # plot_learning_curve(model, "Learning Curve", X_train, y_train, cv=rs)
        # plt.show()
        #return metricModel,score

    def WriteTxt(self, cm, rep, score, metricModel, imgSavePath):
        archi=open("%s.txt" %imgSavePath,'w')
        archi.write('Confusion Matrix \n')
        archi.write('%s \n' %cm)
        archi.write('Other metrics \n')
        archi.write('%s \n' %rep)
        archi.write('Cross Validation \n')
        archi.write('Score\n')
        archi.write('%s \n' %self.score)
        archi.write('Accurancy model \n')
        archi.write('%s \n' %metricModel)
        # archi.write("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
        archi.close()
        print "TXT correctly saved in %s" %imgSavePath
