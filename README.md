# DeadTrees

Algorithm for detection of dead trees in multispectral images.

The work flow is developed in the following steps.


## First step: initialization

The first step is the initialization (initialize.py). It calls the clipshape
functions. The clipshape creates a dictionary, in which the keys are the
categories provided by the training set (shapefile, see detailed description of
the input below), and the values are numpy arrays with shape.
Once this object is created, it is saved into a pickle object in pickle/clip/.
It overwrites, so make sure to chose a different name for the clip if you want a
new one.

## Second step: training

The training is performed by train.py. It takes in input the pickle clip (created
by initialize.py) and the name you want to use for the model. Creates the model
directory pickle/model/ if it doesn't exist. It overwrites, so make sure to
chose a different name for the model if you want a new file.

## Third step: classification
The classification is performed by predict.py. It takes in input the path where
the orthophotos are and loops over them. It also takes as input the folder where
you want to save the results, and the pickle model created by train.py.

## Paths explanation

### RasterPath
Path to training ortophoto.

### Path to shapefile (training set)
The shapefile is a training set, defined by polygons containing different classes.
The classes or categories are indicated in the field "zona" of the attribute
table.
Example:
Category  | Description
1         | Dead trees
2         | Healthy trees
3         | Soil
4         | Shadowed zone
It is important to make sure that all the classes are well represented in the
training set. The training set is created manually.

### Path to ortophotos
Path to ortophotos to be classified.

### Path to pickle model
(Automatically created) where the result of the training is written as a pickle
object.

### Path to pickle clip
(Automatically created) Where the clip created by the initialization should be
written.
