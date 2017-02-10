# DeadTrees

Algorithm for detection of dead trees in multispectral images.

## Input

You need to pass the following input:

### Path to shapefile:
The shapefile is a training set, defined by polygons containing different classes.
The classes or categories are indicated in the field "zona" of the attribute
table.
Example:
Category  | Description
1         | Dead trees
2         | Healthy trees
3         | Soil
4         | Shadowed zone

### Path to ortophotos
Path to ortophotos to be classified.

### Path to pickle model

### Path to pickle clip
Where the clip created by the initialization should be written.

## First step: initialization

The first step is the initialization. It calls the clipshape functions. The
clipshape creates a dictionary, in which the keys are the categories provided by
the training set shapefile, and the values are numpy arrays with shape.
Once this object is created, it is saved into a pickle object in the path to
pickle clip.
