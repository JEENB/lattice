import numpy as np
from src.decompose import *
from src.gaussian_sampling import *

class Lattice:
	def __init__(self, dim: int, basis) -> None:
		self.dim = dim
		self.basis = basis
		if not isinstance(self.basis, np.ndarray):
			raise TypeError("Expected Numpy array")
		elif self.basis.shape != (self.dim, self.dim):
			raise ValueError("Basis Vectors do not match dimension")
		self.grahm_schmidt()

	def grahm_schmidt(self):
		self.orth_basis = None
		self.mu = None
		return self.orth_basis, self.mu


class SampleLattice(Lattice):
	def __init__(self, dim: int, basis, c, sigma:float, tao:float) -> None:
		super().__init__(dim, basis)
		self.c = c
		self.sigma = sigma
		self.tao = tao
		self.orth_basis, self.mu = self.grahm_schmidt()
		self.t = decompose(self.dim, self.orth_basis, self.c)

	def sample(self):
		v = np.zeros(self.dim)
		z = np.zeros(self.dim)
		
		for i in range(self.dim - 1, -1, -1):
			z[i] = DiscreteGaussian(sigma = self.sigma, tao = self.tao, center = self.t[0][i]).sample(1)[0]  ## t is ndarray = [[t_1, t_2, ...]], #sample returns a list so first item
			v = v + z[i] * self.basis[i]





	
		
