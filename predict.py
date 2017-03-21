__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from mlh import *
from serialize import *
from movingwindow import *
# from poligonize import *
#from multiprocessing import Pool

# orthoPath is the path to all ortophotos, whereas rasterPath was only the training set
orthoPath  = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/ortho/"
resultPath = "/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/results_clipfeat5/"

feat = defaultdict(list)
todo = []
count = 0
picklemodel = "modelKNN"
#picklemodel = "RadiusNeighborsC"
model = read("pickle/model/" + str(picklemodel))


Classifier = ImageClassifier(modeltype = 2, \
                             Threads = 4, \
                             picklemodel = picklemodel, \
                             model = model)

# Loops through all orthophotos in the folder
for file in os.listdir(orthoPath):
    if file.endswith(".tif"):
        file = os.path.splitext(file)[0]

        # Classifier.ImageToClassify(orthoPath + str(file) + ".tif", False)
        # False because the indexes don't add information and only confuse the classifier
        # To add more layer, change to True

        Classifier.ImageToClassify(orthoPath + str(file) + ".tif", False)

        Classifier.Classify()
        Classifier.SaveImg(resultPath + str(file) + "_classified")

        # break # uncomment here if you want to see the result on 1 tile

        imgResult = moviw(Classifier.GetClassified(), \
                          resultPath + str(file) + "_smooth", \
                          Classifier.GetProjection(), \
                          Classifier.GetGeotrans())

        print "imgResult", imgResult


        # poligonize(imgResult, resultPath + str(file + "_smooth.tif"))
        # poligonize(resultPath, file)

        count += 1
        if count > 4:
            break

        # if (file in a):
        #file="pt604000-4395000"
        #print file

#p.map(main, todo)
#main(todo[0])
