__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from mlh import *
from serialize import *
from movingwindow import *
from poligonize import *
import re

# orthoPath is the path to all ortophotos
# resultPath is the folder where you want to find the results
# texturePath is the path where the texture layers are
orthoPath  = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/ortho/"
resultPath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/results_texture/"
texturepath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/texture/"

feat = defaultdict(list)
todo = []
count = 0
picklemodel = "modelKNN"
model = read("pickle/model/" + str(picklemodel))


Classifier = ImageClassifier(modeltype = 2, \
                             Threads = 4, \
                             picklemodel = picklemodel, \
                             model = model)

# Loops through all orthophotos in the folder
# builds a dict with name of texture file as key, and location as value
# this dict will be passed to the classifier in order to incorporate the texture layers
# for file in os.listdir(orthoPath):
#     if file.endswith(".tif"):
#         file = os.path.splitext(file)[0]
#         file1 = re.sub("-", "_", file)
#         texturelist = []
#         dict_text = {}
#         for item in os.listdir(texturepath):
#             myregex = "text_" + file1
#             if item.startswith(myregex) :
#                 item1 = item.split(".")[0]
#                 texturelist.append(item1)
#                 dict_text.setdefault(item1, []).append(orthoPath + item)
#         # print dict_text


for file in os.listdir(orthoPath):
    if file.endswith(".tif"):
        file = os.path.splitext(file)[0]

        # Classifier.ImageToClassify(orthoPath + str(file) + ".tif", True, **dict_text)
        Classifier.ImageToClassify(orthoPath + str(file) + ".tif", True, texturepath)

        Classifier.Classify()
        Classifier.SaveImg(resultPath + str(file) + "_classified")

        # break # uncomment here if you want to see the result on 1 tile

        imgResult = moviw(Classifier.GetClassified(), \
                          resultPath + str(file) + "_smooth", \
                          Classifier.GetProjection(), \
                          Classifier.GetGeotrans())

        print "imgResult", imgResult

        # TODO: fix poligonize function
        # poligonize(imgResult, resultPath + str(file + "_smooth.tif"))
        # poligonize(resultPath, file)

        count += 1
        if count > 4:
            break



        # True because added texture layers
        #
        # Classifier.Classify()
        # Classifier.SaveImg(resultPath + str(file) + "_classified")
        #
        # # break # uncomment here if you want to see the result on 1 tile
        #
        # imgResult = moviw(Classifier.GetClassified(), \
        #                   resultPath + str(file) + "_smooth", \
        #                   Classifier.GetProjection(), \
        #                   Classifier.GetGeotrans())
        #
        # print "imgResult", imgResult
        #
        # # TODO: fix poligonize function
        # # poligonize(imgResult, resultPath + str(file + "_smooth.tif"))
        # # poligonize(resultPath, file)
        #
        # count += 1
        # if count > 4:
        #     break
