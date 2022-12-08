import numpy as np
from decompose import *
from gaussian_sampling import *
from grahm_schmidt import *

class Lattice:
	def __init__(self, dim: int, basis) -> None:
		self.dim = dim
		self.basis = basis
		if not isinstance(self.basis, np.ndarray):
			raise TypeError("Expected Numpy array")
		elif self.basis.shape != (self.dim, self.dim):
			raise ValueError("Basis Vectors do not match dimension")
		self.mu, self.orth_basis = grahm_schmidt(dim=self.dim, basis=self.basis)
		self.ri = [np.linalg.norm(i) for i in self.orth_basis]


class SampleLattice(Lattice):
	def __init__(self, dim: int, basis, c, sigma:float, tao:float) -> None:
		super().__init__(dim, basis)
		self.c = c
		self.sigma = sigma
		self.tao = tao
		self.t = decompose(self.dim, self.orth_basis, self.c)

	def sample(self):
		v = np.zeros(self.dim, dtype=float)
		z = np.zeros(self.dim, dtype=float)
		
		for i in range(self.dim - 1, -1, -1):
			z[i] = DiscreteGaussian(sigma = self.sigma/self.ri[i], tao = self.tao, center = self.t[0][i]).sample(1)[0]  ## t is ndarray = [[t_1, t_2, ...]], #sample returns a list so first item
			v = v + z[i] * self.basis[i]
			self.t = self.t - z[i]*self.mu[i]
		return v

s = SampleLattice(dim=3, basis = np.array([[0,1,2], [1,4,0], [3,0,4]], dtype=float), c= np.array([[1,1,1]],dtype=float), sigma=5, tao=1).sample()
print(s)



	
		
