''' linear algebra utils'''
import numpy.linalg as linalg
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

MIN = -99999999999999
MAX = 999999999999999

def dot(vec1:list, vec2:list) -> int:
	'''
	Computes the dot product between two vectors. 
	np.dot may give errors
	'''
	if len(vec1) != len(vec2):
		raise ValueError(f"Vec1 and Vec2 are of different length.")
	s = 0
	for i in range(len(vec1)):
		s += vec1[i]*vec2[i]
	return s
	
def norm(vector:list):
	'''
	returns the eucledian norm of the vector
	'''
	return linalg.norm(vector)


def check_via_graph(vector: list):
	'''
	creates a frequency distribution graph from a list
	if list is given
	'''
	sns.distplot(vector)
	plt.show()