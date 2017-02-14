__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

import pickle

def save(pathpickleclip, Mylist):
	'''This function uses pickle to convert a python object in RAM into a
	character stream (file)
    '''
	with open(pathpickleclip + '.pickle', 'wb') as handle:
		pickle.dump(Mylist, handle)


def read(pathpickleclip):
	'''This function uses pickle to load in RAM a python object stored as a
 	character stream
	'''
	with open(pathpickleclip + '.pickle', 'rb') as handle:
		Mylist = pickle.load(handle)
	return Mylist
