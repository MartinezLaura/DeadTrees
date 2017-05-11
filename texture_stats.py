__author__ = "Laura Martinez Sanchez, Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "2.0"
__email__ = "lmartisa@gmail.com, dileomargherita@gmail.com"



from clipshape import *
import itertools
from mlh import *
from collections import defaultdict
import numpy as np
from texture_initialize import *
import matplotlib.pyplot as plt

'''
This code is used to establish the variability expressed by each texture layer
in each class
'''


feat = defaultdict(list)
field = 'zona' # field in the shapefile where to read the classes
# The following is a shapefile with polygons representing the various classes
shapePath = '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/Features/mosaic5.shp'
textpath= '/home/v-user/shared/Documents/Documents/CANHEMON/classification_tests/texture_training/'

INX = False # Classification based on only 4 bands




#----------------------------------------------------------------------

def myboxplot(title, data, median, deviation, categories):

    path = "/home/v-user/shared/Documents/Documents/CANHEMON/texture_stats/"

    fig = plt.figure()

    plt.boxplot(data, notch = True)
    # plt.xticks([1, 2, 3, 4], categories)
    plt.xticks([1, 2, 3, 4], ["Dead Tree", "Soil", "Healthy Tree", "Shadow"])
    plt.title(title)

    fig.savefig(path + title + ".png")

    # plt.show()



#----------------------------------------------------------------------

if __name__ == "__main__":

    for file in os.listdir(textpath):
        if file.endswith(".tif"):
            file = os.path.splitext(file)[0]
            print file
            # init_texture(field, textpath + str(file)+ ".tif", shapePath, INX, file)

            # file = "text_training_mosaic_b1_ASM"

            with open('pickle/clip/' + str(file) + '.pickle', 'rb') as handle:
            	Mylist = pickle.load(handle)

            feat = Mylist[0]
            print "feat.keys()", feat.keys()

            temp = defaultdict(list).fromkeys(feat)
            # print 'temp:', temp
            categories = []
            median = []
            deviation = []
            data = []

            for key, value in feat.iteritems():

                texturefile = str(file)
                temp[str(key)] = np.concatenate(value)
                element = [texturefile, key, value, temp[str(key)]]

                print "Texture file: ", texturefile
                print "Category: ", key
                categories.append(key)
                print "Median value: ", np.median(temp[str(key)])
                median.append(np.median(temp[str(key)]))
                print "Deviation: ", np.std(temp[str(key)])
                a = np.std(temp[str(key)])
                deviation.append((-a, a))

                data.append(temp[str(key)])


            print deviation
            myboxplot(file, data, median, deviation, categories)





                # print "---------Texture used" + str(file) + "----------"
                # print "---------Classification type " + str(key) + "----------"
                # deviation = np.std(temp[str(key)] , axis = 0)
                # mean = np.mean(temp[str(key)] , axis = 0)
                # print "deviation", deviation
                # print "mean", mean
