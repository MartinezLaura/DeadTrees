__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



# from clipshape import *
import itertools
from mlh import *
from collections import defaultdict
import numpy as np
from serialize import *
from graficas import *

pickleclip  = "clipfeat-5"
nLayers = 4 #number of layers


# Use this to calculate the numbers to put in graph in presentation 2 page 18
# The following code is to evaluate the performance of the method.


def Metrics(arrTrue, arrPredict, labels):
    cm = metrics.confusion_matrix(arrTrue, arrPredict, labels = map(int, labels))
    rep = metrics.classification_report(arrTrue, arrPredict)
    print rep
    print cm
    return cm, rep

def CrossValidation(X, y, model, labels):
    scores = np.empty(shape = (1, 3))
    averages = np.empty(shape = (1, 3))
    kf = KFold(len(y), n_folds = 3)
    # KFold splits dataset into k consecutive folds (without shuffling by default)
    skf = StratifiedKFold(y, 10)
    # StratifiedKFold is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    rs = cross_validation.ShuffleSplit(len(y), \
                                       n_iter = 100, \
                                       test_size = .2, \
                                       random_state = 0)
    # Random permutation cross-validation iterator.
    # Yields indices to split data into training and test sets.
    # Note: contrary to other cross-validation strategies, random splits do not
    # guarantee that all folds will be different, although this is still very
    # likely for sizeable datasets.

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
        # print "train : ", train
        # print "test : ", test
        # print "Model : ", model
        # print "Labels : ", labels
        # print "Score : ", score
        # print "Metric Model : ", metricModel
        scores[inx] = score
        averages[inx] = metricModel
        inx += 1

    metricModel = np.average(averages)
    score = np.average(scores)
    print "X_train : ", X_train
    print "y_train : ", y_train
    print "Score : ", score
    print "Metric Model : ", metricModel
    plot_learning_curve(model, "Learning Curve", X_train, y_train)
    plt.show()
    return metricModel, score


Mylist = read("pickle" + os.sep + "clip" + os.sep + pickleclip)

feat = Mylist[0]
nPixels = Mylist[1]

X = np.empty((nPixels, nLayers), dtype = float)
y = np.empty((nPixels), dtype = np.uint8)
	#w = np.empty((nPixels),dtype=float)

offset = 0
for i in feat:
    for j in feat.get(i):
        X[offset : offset + j.shape[0], :] = j
        y[offset : offset + j.shape[0]] = i
        # 	if i == '0':
        # 		w[offset:offset+j.shape[0]] = 2000000
        # else:
        # 	w[offset:offset+j.shape[0]] = 1
        offset += j.shape[0]

#Metrics
model = KNeighborsClassifier(n_neighbors = 5, \
                             weights = 'distance', \
                             algorithm = 'ball_tree', \
                             leaf_size = 100, \
                             p = 1, \
                             metric = 'minkowski', \
                             metric_params = None, \
                             n_jobs = 8)
model.fit(X, y)
cm, rep = Metrics(y, model.predict(X), sorted(feat.keys()))
metricModel, score = CrossValidation(X, y, model, sorted(feat.keys()))
print "---------cm--------------------"
print cm
print "---------rep-------------------"
print rep
print "---------metric model----------"
print metricModel
print "---------score-----------------"
print score
