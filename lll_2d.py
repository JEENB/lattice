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


class GramSchmidt(Basis):
	def gram_sch(self, steps = False):
		inter_result = []
		m = dot(self.v_0, self.v_1)/norm(self.v_0)**2
		v_1 = np.subtract(self.v_1, np.multiply(m, self.v_0))
		inter_result.append((self.v_0, v_1))
	
		if steps == True:
			print(tabulate(inter_result, headers= ['v_1', 'v_2'], tablefmt="pretty", numalign='center') )
					
		return self.v_0, v_1

b = Basis(v_0= [66586820, 65354729],v_1= [6513996, 6393464])
b1 = Basis(v_0= [66586820, 65354729],v_1= [6513996, 6393464])

print("LLL Reduction")
LLL2D.lll_dim2(b, steps=True)

print("Gram Schmidt Ortho")
GramSchmidt.gram_sch(b1, steps=True)
