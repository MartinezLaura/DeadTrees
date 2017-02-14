__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from mlh import *
from serialize import *
from movingwindow import *
from poligonize import *
#from multiprocessing import Pool

# orthoPath is the path to all ortophotos, whereas rasterPath was only the training set
#orthoPath = "/home/v-user/canhemon/H03_CANHEMON/Imagery/Portugal/DMC/ortophotos_05022016/"
orthoPath  = "/home/v-user/canhemon/H03_CANHEMON/test_madi/ortho/"
resultPath = "/home/v-user/canhemon/H03_CANHEMON/test_madi/"

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
for file in os.listdir(orthoPath):
    if file.endswith(".tif"):
        file = os.path.splitext(file)[0]
        # if not os.path.isfile('resultPath'+file + ".shp"):

        Classifier.ImageToClassify(orthoPath + str(file) + ".tif", False)
        # False because the indexes don't add information and only confuse the classifier

        Classifier.Classify()
        Classifier.SaveImg(resultPath + str(file) + "_classified")

        # break # uncomment here if you want to see the result on 1 tile
        #imgResult = read("pickle"+os.sep+"images"+os.sep+nameresult)

        imgResult = moviw(Classifier.GetClassified(), \
                          resultPath + str(file) + "_smooth", \
                          Classifier.GetProjection(), \
                          Classifier.GetGeotrans())

        print "imgResult", imgResult

        # TODO: fix poligonize function
        # poligonize(imgResult, resultPath + str(file + "_smooth.tif"))

        count += 1
        if count > 4:
            break

        # if (file in a):
        #file="pt604000-4395000"
        #print file

#p.map(main, todo)
#main(todo[0])
