__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from mlh import *
from serialize import *

#from multiprocessing import Pool

# the pickle of the model is different from the clipshape
picklemodel = "modelKNN" # name
#picklemodel = "RadiusNeighborsC"

# here you can put None to create a new pickle, or the name of an existing pickle
# For example: the pickle created by initialize.py
pickleclip  = "clipfeat-5-text"

layer = 4 # 4 bands
Classifier = ImageClassifier(modeltype = 2, \
                             Threads = 4, \
                             picklemodel = picklemodel, \
                             model = None)
Mylist = read("pickle" + os.sep + "clip" + os.sep + pickleclip)
feat = Mylist[0]
nPixels = Mylist[1]
print "nPixels", nPixels

# if picklemodel == None:
#     MyName = pickleclip + "model"
# else:
#     MyName = picklemodel

print "feat.keys()", feat.keys()
Classifier.Train(feat, nPixels, layer, picklemodel)

mod = read("pickle/model/" + str(picklemodel))
print "Model: ", mod
