import numpy as np
from exception import *

class Lattice:
	def __init__(self, dim: int, basis) -> None:
		self.dim = dim
		self.basis = basis
		
		if not isinstance(self.basis, np.ndarray):
			raise TypeError("Basis not a numpy array")
		if self.basis.shape != (self.dim, self.dim):
			raise ShapeMismatch(f"Expected ({self.dim, self.dim}) got {self.basis.shape}")

		if np.linalg.matrix_rank(self.basis) != self.dim:
			raise NotIndependentVectors("The given basis are not independent")
		
		self.hadamard_ratio = self.__hadamard_ratio()
		self.__grahm_schmidt()

	def __hadamard_ratio(self):
		'''
		Computes the hadamard_ratio
		'''
		b = self.basis
		self.det = np.linalg.det(b)
		self.vol = 1
		for i in b:
			self.vol *= np.linalg.norm(i)
		return abs(self.det/self.vol) ** (1/self.dim)


	def __grahm_schmidt(self):
		'''
		Performs the grahm-schmidt orthogonalization. 
		'''
		self.orth_basis = self.basis.astype(float)
		self.mu = np.identity(self.dim)
		for i, vec in enumerate(self.orth_basis):
			s = np.zeros(self.dim, dtype=float)
			for j in range(i):
				m = np.dot(vec, self.orth_basis[j])/np.linalg.norm(self.orth_basis[j])**2
				self.mu[i,j] = m
				s += m * self.orth_basis[j]
			self.orth_basis[i] = np.subtract(self.orth_basis[i], s)

	def hadamard_ratio(self, basis):
		'''
		Computes the hadamard_ratio
		'''
		det = np.linalg.det(basis)
		vol = 1
		dim = np.linalg.matrix_rank(basis)
		for i in basis:
			vol *= np.linalg.norm(i)
		return abs(det/vol) ** (1/dim)

	def generate_bad_basis(self):
		U = np.array([[4327, 3297, 5464], [-15447, -11770, -19506], [23454, 17871, 29617]]) ## replace with matrix det 1 generation function
		return np.matmul(U.T, self.basis)



