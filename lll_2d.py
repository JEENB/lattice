'''
2 Dimension lattice reduction. Implemented as per the algorithm described in Proposition 6.63 from Introduction to Mathematical Cryptography. 
'''
from lattice_code.utils import *
import numpy as np
import numpy.linalg as linalg
from tabulate import tabulate

class LLL2D:
	def __init__(self, v_0:list, v_1:list, dim = 2) -> None:
		self.dim = dim
		self.v_0 = v_0
		self.v_1 = v_1

		if len(self.v_0) != len(self.v_0):
			raise ValueError(f"Vector length mismatch.")

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


b = LLL2D(2, v_0= [66586820, 65354729],v_1= [6513996, 6393464])
b.lll_dim2(steps=True)