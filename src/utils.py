''' linear algebra utils'''
import numpy.linalg as linalg
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import itertools

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


def frequency_plot(vector: list, title: str = None, x_label: str = None, y_label: str = None):
	'''
	creates a frequency distribution graph from a list
	if list is given
	'''

	frequency = dict(collections.Counter(vector))
	print(frequency)
	plt.bar(x = frequency.keys(), height = frequency.values(), linewidth=0.5)
	# sns.histplot(vector, bins=max(vector)+1)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()


class Interval:

	def __init__(self, a:float, b:float) -> None:

		'''
		a is the lower limit and b it the upper limit
		'''
		self.a = a
		self.b = b

	def check(self, c:float)-> bool:
		'''checks if c is in the interval or not'''
		if c >= self.a and c <= self.b:
			return True 
		else: 
			return False
		
	def print_interval(self, title = None):
		print(f"{title}", (self.a, self.b))


def generate_all_possible_sequence(q, L):
	lower = -int(q/2)
	upper = int(q/2)
	subset = itertools.product(range(lower, upper + 1),  repeat = L)
	return subset
