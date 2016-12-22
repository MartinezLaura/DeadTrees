__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"


from clipshape import *
import itertools
from mlh import *
from collections import defaultdict
import numpy as np



feat = defaultdict(list)
# with open('pickle/clipshapes4inx.pickle', 'rb') as handle:
# 	Mylist = pickle.load(handle)

# feat = Mylist[0]
# print feat.keys()

# ##Calculo de las desviaciones y medias
# temp = defaultdict(list).fromkeys(feat)


# for key, value in feat.iteritems():
# 	temp[str(key)] = np.concatenate(value)
# 	print "---------"+str(key)+"----------"
# 	deviation = np.std(temp[str(key)] , axis = 0)
# 	mean = np.mean(temp[str(key)] , axis = 0)
# 	print deviation
# 	print mean


#### para el metodo


with open('pickle/clipshapes4.pickle', 'rb') as handle:
	Mylist = pickle.load(handle)

feat = Mylist[0]
nPixels = Mylist[1] 

X = np.empty((nPixels,4),dtype=float)
y = np.empty((nPixels),dtype=np.uint8)
	#w = np.empty((nPixels),dtype=float)

offset=0
for i in feat: 
	for j in feat.get(i):
	 	X[offset:offset+j.shape[0],:] = j
	 	y[offset:offset+j.shape[0]] = i
	 	# 	if i == '0':
	 	# 		w[offset:offset+j.shape[0]] = 2000000
			# else:
			# 	w[offset:offset+j.shape[0]] = 1
	 	offset+=j.shape[0]

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
