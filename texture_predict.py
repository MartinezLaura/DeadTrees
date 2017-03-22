__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from mlh import *
from serialize import *
from movingwindow import *
import os

import re

# orthoPath is the path to all ortophotos
# resultPath is the folder where you want to find the results
# texturePath is the path where the texture layers are
orthoPath  = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/ortho/"
texturepath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/texture/"
resultPath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/results_texture/"


feat = defaultdict(list)
todo = []
count = 0
picklemodel = "modelKNN"
model = read("pickle/model/" + str(picklemodel))


Classifier = ImageClassifier(modeltype = 2, \
                             Threads = 4, \
                             picklemodel = picklemodel, \
                             model = model)

for file in os.listdir(orthoPath):
    if file.endswith(".tif"):
        file = os.path.splitext(file)[0]


        Classifier.ImageToClassify(orthoPath + str(file) + ".tif", True, texturepath)
        # True because added texture layers

        Classifier.Classify()
        Classifier.SaveImg(resultPath + str(file) + "_classified")

        # break # uncomment here if you want to see the result on 1 tile

        imgResult = moviw(Classifier.GetClassified(), \
                          resultPath + str(file) + "_smooth", \
                          Classifier.GetProjection(), \
                          Classifier.GetGeotrans())

        print "imgResult", imgResult

        count += 1
        if count > 4:
            break
