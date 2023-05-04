import numpy as np
from sampling import *
from gram_schmid import *
from utils import *
from lattice import *

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
			raise ValueError(f"Vector shape Mismatch {orth_basis}")
	else:
		raise TypeError("Expected numpy array")

	ri_square = np.array([[norm(i)**2 for i in orth_basis]])
	y = np.matmul(c, orth_basis)
	return np.true_divide(y,ri_square)


class SampleLattice(Lattice):
	def __init__(self, dim: int, basis, c, sigma:float, tao:float) -> None:
		super().__init__(dim, basis)
		self.ri = [np.linalg.norm(i) for i in self.orth_basis]
		self.c = c
		self.sigma = sigma
		self.tao = tao

		if np.linalg.matrix_rank(self.basis) != self.dim: 
			raise NotIndependentVectors("The basis vectors are not linearly independent. Try different vector or use generate_lattice_points")

	def sample(self):
		v = np.zeros(self.dim, dtype=np.longdouble)
		z = np.zeros(self.dim, dtype=np.longdouble)
		t = decompose(self.dim, self.orth_basis, self.c)
		
		for i in range(self.dim - 1, -1, -1):
			z[i] = DiscreteGaussian(sigma = self.sigma/self.ri[i], tao = self.tao, center = t[0][i]).sample(1)[0]  ## t is ndarray = [[t_1, t_2, ...]], #sample returns a list so first item
			v = v + z[i] * self.basis[i]
			t = t - z[i]*self.mu[:,i]
		return v.astype('int')
	
	def verify_point(self, point):
		s = np.linalg.solve(self.basis, point)
		for i in range(self.dim):
			if abs(s[i]).is_integer() == False:
				return False
			else:
				pass
		return True



if __name__ == '__main__':

	s = SampleLattice(dim=3, basis = np.array([[0,0,1], [0,1,0], [1,0,0]]), c= np.array([[1,5,9]]), sigma=5, tao=3)
	print(s.hadamard_ratio)
	samples = s.sample()
	print(samples)
	print(s.verify_point(samples))

	

	
		
