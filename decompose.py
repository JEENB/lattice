import numpy as np
from utils import *


class Decompose:

	def __init__(self, n:int, c , B ):
		'''
		Decompose a vector c over the GS basis
		Parameters
		-----------
		n : int
			lattice dimension
		
		c : numpy.array of dimension (1, n)
			vector eg: np.array([[1,2,3]])
		
		B : numpy.array of dimension (1, n)
			basis vector of lattice
			eg: np.array([	
						[0,1,0]: b_1,
						[1,0,0]: b_2
						[0,0,1]: b_3
						])
		'''

		self.n = n
		self.c = c
		self.B = B
		self.ri = np.array([[norm(i)**2 for i in self.B]])

		if isinstance(self.c, np.ndarray) and isinstance(self.B, np.ndarray) and isinstance(self.ri, np.ndarray):
			if self.c.shape != (1, n) or self.B.shape != (n,n) or self.ri.shape != (1,n):
				raise ValueError("Vector shape Mismatch")
		else:
			raise TypeError("Expected numpy array")

	def decompose(self):
		'''
		Compute
		-------
		y = c. B  #paper suggests B*t but since bi's are already row vectors transpose is not requuired
		return (y_i/r_i^2)
		'''
		self.y = np.matmul(self.c, self.B)
		return np.true_divide(self.y,self.ri)

		

		


d = Decompose(2, np.array([[1,2]]), np.array([[0,1],[1,0]]), np.array([[1,1]]))
