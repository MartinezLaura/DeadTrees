__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

import pickle

def save(pathpickleclip, Mylist):
	with open(pathpickleclip+'.pickle', 'wb') as handle:
		pickle.dump(Mylist, handle)


def read(pathpickleclip):
	with open(pathpickleclip+'.pickle', 'rb') as handle:
		Mylist = pickle.load(handle)
	return Mylist