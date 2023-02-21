import numpy as np
from exception import *
from sampling import *

'''TODO: Custom parameter selection need a class to export parameter selection'''
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


class LWE:
	'''
	LWE instance generation class
	Parameters
	-----------
	q: int
		group size, Z_q
	n: int
		size of the column vector
	m: int
		number of LWE instance generations
	alpha: float
		parameter for discrete gaussian, \sigma = \alpha * q
	secret: list
		vector of length n sampled from a distribution
	
	Algorithm
	-----------
	for i in 1 to m:
		a_i <-$ Z_q^n
		e_i <-D_\sigma = alpha*q
		b_i = a_i^T*s + e_i
	output: {a_i, b_i} for i in 1 to m
	'''
	def __init__(self, q: int, n: int, alpha:float, error_sampling: str = 'Gaussian', secret_sampling: str = 'Uniform'):
		self.q = q
		self.n = n
		self.alpha = alpha
		self.secret = self.__generate_secret()
		self.error_sampling = error_sampling
		self.secret_sampling = secret_sampling
		self.e = []

	def __generate_secret(self):
		'''
		Generates a secret of length n depending on the distribuiton. 
		Parameters:
		-------------
		distribution: Gaussian/ Uniform
			Default: Gaussian
		'''
		return uniform_sampling(0, self.q, self.n)


	def an_instance(self):
		'''Generates one instance of LWE'''
		a_i = uniform_sampling(0, self.q, self.n)
		e_i = discrete_gaussian_sampling(0, sigma= self.alpha*self.q, tao= 3, sample_points=1)
		self.e.append(e_i)
		b_i = (np.dot(a_i, self.secret) + e_i) % self.q
		return a_i, b_i


	def LWE_instances(self, m):
		'''Generates m LWE instances'''
		instances = []
		for i in range(m):
			a_i, b_i = self.an_instance()
			instances.append((a_i, b_i))
		return instances


	def LWE_instances_matrix_version(self, m):
		'''Matrix version of the LWE instances A, B'''
		A = np.zeros(shape = (self.n, m))
		b = []
		for i in range(m):
			a_i, b_i = self.an_instance()
			A[:,i] = np.array(a_i)
			b.append(b_i)
		return A, b


l = LWE(q = 100, n = 10, m = 2, alpha= 0.1)
print(l.LWE_instances_matrix_version())
print(l.secret)
print(l.e)
