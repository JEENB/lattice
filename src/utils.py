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



def plot_lattice_2d(a,b,a_,b_, points_count = 10):
    '''
	Creates a plot of lattice, with good and bad basis together
	'''
    points_count= points_count
    points=[a,b]
    for x in range(-points_count//2+1,points_count//2+1):
        for y in range(-points_count//2+1,points_count//2+1):
            points.append(a*x+b*y)
    points=np.array(points)
    
    points_=[a_,b_]
    for x in range(-points_count//2+1,points_count//2+1):
        for y in range(-points_count//2+1,points_count//2+1):
            points_.append(a_*x+b_*y)
    points_=np.array(points_)
    
    fig, ax = plt.subplots()
    
    ax.scatter(x=points[:,0],y=points[:,1],c="red")
    ax.scatter(x=points_[:,0],y=points_[:,1],c="blue")
    ax.scatter(x=[0],y=[0])
    
    ax.plot([0,a[0]],[0,a[1]],color="r",label ="original basis")
    ax.plot([0,b[0]],[0,b[1]],color="r")
    ax.plot([0,a_[0]],[0,a_[1]],color="b",label ="reduced basis")
    ax.plot([0,b_[0]],[0,b_[1]],color="b")
    
    ax.minorticks_on()
    ax.grid(which='minor',
        color = 'black',
        linewidth = 0.1)
    ax.legend()
    ax.grid()
    plt.show() 

def generate_lattice_points(dimension, q):
	'''
	q: parameter for finite field Z_q
	dimension: lattice dimension
	generates a (dimension * dimension) matrix with random entries from Z_q. The basis can be any
	'''
	det = 0
	while det == 0:
		M = np.random.randint(-q, q, size=(dimension, dimension))
		det = np.linalg.det(M)
	return M