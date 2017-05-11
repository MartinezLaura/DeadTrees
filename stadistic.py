__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



# from clipshape import *
import itertools
from mlh import *
from collections import defaultdict
import numpy as np


feat = defaultdict(list)


with open('pickle/clip/clipfeat-5-text_1.pickle', 'rb') as handle:
	Mylist = pickle.load(handle)

feat = Mylist[0]
print feat.keys()

##Calculate mean and st. dev.
temp = defaultdict(list).fromkeys(feat)


for key, value in feat.iteritems():
	temp[str(key)] = np.concatenate(value)
	print "---------" + str(key) + "----------"
	deviation = np.std(temp[str(key)], axis = 0)
	mean = np.mean(temp[str(key)], axis = 0)
	print deviation
	print mean
#----------------------------------------------------------------------
# Use this to calculate the numbers to put in graph in presentation 2 page 18
#### para el metodo
# The following code is to evaluate the performance of the method.

#
#  def Metrics(self,arrTrue, arrPredict, labels):
#    cm = metrics.confusion_matrix(arrTrue, arrPredict, labels = map(int,labels))
#    rep = metrics.classification_report(
# 	   arrTrue,arrPredict)
#    print rep
#    print cm
#    # return cm,rep
#
#  def CrossValidation(self,X,y,model,labels):
#    scores = np.empty(shape=(1,3))
#    averages = np.empty(shape=(1,3))
#    #kf = KFold(len(y), n_folds=3)
#    skf = StratifiedKFold(y, 10)
#    #rs = cross_validation.ShuffleSplit(len(y), n_iter=100,test_size=.2, random_state=0)
#    for train, test in skf:
# 	   inx = 0
# 	   X_train, X_test = X[train], X[test]
# 	   y_train, y_test = y[train], y[test]
# 	   predicted = model.fit(X_train,y_train)
# 	   y_pred = predicted.predict(X_test)
# 	   Metrics(y_test, y_pred, labels)
# 	   score = predicted.score(X_test,y_test)
# 	   predicted = cross_validation.cross_val_predict(model, X_train,y_train)
# 	   metricModel = metrics.accuracy_score(y_train, predicted)
# 	   print score
# 	   print metricModel
# 	   scores[inx] = score
# 	   averages[inx] = metricModel
# 	   inx += 1
#    self.metricModel = np.average(averages)
#    self.score = np.average(scores)
#    # plot_learning_curve(model, "Learning Curve", X_train, y_train, cv=rs)
#    # plt.show()
#    #return metricModel,score
#
#
# with open('pickle/clipshapes4.pickle', 'rb') as handle:
# 	Mylist = pickle.load(handle)
#
# feat = Mylist[0]
# nPixels = Mylist[1]
#
# X = np.empty((nPixels,4),dtype=float)
# y = np.empty((nPixels),dtype=np.uint8)
# 	#w = np.empty((nPixels),dtype=float)
#
# offset=0
# for i in feat:
# 	for j in feat.get(i):
# 	 	X[offset:offset+j.shape[0],:] = j
# 	 	y[offset:offset+j.shape[0]] = i
# 	 	# 	if i == '0':
# 	 	# 		w[offset:offset+j.shape[0]] = 2000000
# 			# else:
# 			# 	w[offset:offset+j.shape[0]] = 1
# 	 	offset+=j.shape[0]

#Metrics
#model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma=1/0.2, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
# model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=100, p=1, metric='minkowski', metric_params=None, n_jobs=8)
# # model.fit(X, y)
# # cm,rep=Metrics(y, model.predict(X), sorted(feat.keys()))
# metricModel,score = CrossValidation(X,y,model,sorted(feat.keys()))
# # print "---------cm----------"
# print cm
# print "---------rep----------"
# print rep
# print "---------metric model----------"
# print metricModel
# print "---------score----------"
# print score
