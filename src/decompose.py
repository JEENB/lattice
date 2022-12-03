import numpy as np
from src.utils import *


def decompose(dim, orth_basis, c):
	'''
	Decompose a vector c over the GS orth_basis
	Parameters
	-----------
	n : int
		lattice dimension
	
	c : numpy.array of dimension (1, n)
		vector eg: np.array([[1,2,3]])
	
	B : numpy.array of dimension (1, n)
		orth_basis vector of lattice
		eg: np.array([	
					[0,1,0]: b_1,
					[1,0,0]: b_2
					[0,0,1]: b_3
					])
	Compute
	-------
	y = c. B  #paper suggests B*t but since bi's are already row vectors transpose is not requuired
	return (y_i/r_i^2)
	'''

	if isinstance(c, np.ndarray) and isinstance(orth_basis, np.ndarray):
		if c.shape != (1, dim) or orth_basis.shape != (dim, dim):
			raise ValueError("Vector shape Mismatch")
	else:
		raise TypeError("Expected numpy array")

	ri_square = np.array([[norm(i)**2 for i in orth_basis]])
	y = np.matmul(c, orth_basis)
	return np.true_divide(y,ri_square)

	

print(decompose(2, np.array([[0,1],[1,0]]),np.array([[1,2]])))
