'''
2 Dimension lattice reduction. Implemented as per the algorithm described in Proposition 6.63 from Introduction to Mathematical Cryptography. 
'''
from utils import *
import numpy as np
import numpy.linalg as linalg
from tabulate import tabulate


class Basis:
	def __init__(self, v_0:list, v_1:list, dim = 2) -> None:
		self.dim = dim
		self.v_0 = v_0
		self.v_1 = v_1

		if len(self.v_0) != len(self.v_0):
			raise ValueError(f"Vector length mismatch.")

class LLL2D(Basis):

	def __init__(self, v_0: list, v_1: list, dim=2) -> None:
		super().__init__(v_0, v_1, dim)
	

	def lll_dim2(self, steps = False):
		inter_result = []
		m = np.inf
		i = 0
		while m != 0:
			if norm(self.v_1) < norm(self.v_0):
				self.v_0, self.v_1 = self.v_1, self.v_0
			m = np.round_(dot(self.v_0, self.v_1)/norm(self.v_0)**2)
			self.v_1 = np.subtract(self.v_1, np.multiply(m, self.v_0))
			i += 1
			inter_result.append((i, m, self.v_0, self.v_1))
		
		if steps == True:
			print(tabulate(inter_result, headers= ['Step', 'm', 'v_1', 'v_2'], tablefmt="pretty", numalign='center') )
					
		return i, m, self.v_0, self.v_1



def gram_sch_2D(v_0, v_1, steps = False):
	inter_result = []
	m = dot(v_0, v_1)/norm(v_0)**2
	v_1 = np.subtract(v_1, np.multiply(m, v_0))
	inter_result.append((v_0, v_1))

	if steps == True:
		print(tabulate(inter_result, headers= ['v_1^*', 'v_2^*'], tablefmt="pretty", numalign='center') )		
	return v_0, v_1
	
if __name__ == '__main__':
	b = generate_lattice_points(2, 1024)
	b = np.array([[12345, 1234], [5678, 1456]])
	print(b)

	v_0 ,v_1 =  b[0,:], b[1,:]

	print("LLL Reduction")
	v_0 = [12345, 1234]
	v_1 = [5678, 1456]
	lll = LLL2D(v_0, v_1)
	_, __, v_0star, v_1star = lll.lll_dim2(steps=True)
	print(v_0star, v_1star)

	print("Gram Schmidt Ortho")
	grahm_schmidt_basis = gram_sch_2D(v_0, v_1, steps=False)
	print(grahm_schmidt_basis)

	plot_lattice_2d(v_0, v_1, v_0star, v_1star, points_count=7)

