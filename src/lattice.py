import numpy as np
from sampling import *
from tabulate import tabulate
from gram_schmid import gram_schmid
'''TODO: Custom parameter selection need a class to export parameter selection'''

class ShapeMismatch(Exception):
	'''Exception when shape of the matrix mismatches.'''
	pass

class NotIndependentVectors(Exception):
	'''Exception when dimension'''
	pass

def rand_unimod(n):
	'''
	Generates a unimodular matrix for bad basis generation. 
	Code from: https://github.com/PizzaEnthusiast/GGH-CryptoSystem/blob/master/ggh.py
	'''
	random_matrix = [ [np.random.randint(-10000,10000,) for _ in range(n) ] for _ in range(n) ]
	upperTri = np.triu(random_matrix,0)
	lowerTri = [[np.random.randint(-10,10) if x<y else 0 for x in range(n)] for y in range(n)]  

	#we want to create an upper and lower triangular matrix with +/- 1 in the diag  
	for r in range(len(upperTri)):
		for c in range(len(upperTri)):
			if r == c: 
				if bool(random.getrandbits(1)):
					upperTri[r][c] = 1
					lowerTri[r][c] = 1
				else:
					upperTri[r][c] = -1
					lowerTri[r][c] = -1
	uni_modular = np.matmul(upperTri, lowerTri)
	return uni_modular


class Lattice:
	'''
	represent basis vectors as row vectors
	'''
	def __init__(self, dim: int, basis: np.ndarray) -> None:
		self.dim = dim
		self.basis = basis
		
		if not isinstance(self.basis, np.ndarray):
			raise TypeError("Basis not a numpy array")
		if self.basis.shape != (self.dim, self.dim):
			raise ShapeMismatch(f"Expected ({self.dim, self.dim}) got {self.basis.shape}")

		# if np.linalg.matrix_rank(self.basis) != self.dim:
		# 	raise NotIndependentVectors("The given basis are not independent")
		
		self.hadamard_ratio = self.__hadamard_ratio()
		# print("Hadamard Ratio = ", self.hadamard_ratio)
		self.orth_basis, self.mu = gram_schmid(self.basis)
		
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
	
	@staticmethod
	def had_ratio(basis:np.ndarray):
		'''
		Computes the hadamard_ratio
		'''
		det = np.linalg.det(basis)
		dim = basis.shape[0]
		vol = 1
		for i in basis:
			vol *= np.linalg.norm(i)
		return abs(det/vol) ** (1/dim)

	@staticmethod
	def generate_bad_basis(basis:np.ndarray):
		'''
		generates a bad basis given a good basis using the unimod matrix
		'''
		U = rand_unimod(basis.shape[1])
		bad_basis = np.matmul(U, basis)
		return bad_basis
	
	@staticmethod
	def babais_cvp(basis:np.ndarray,target_vector:np.ndarray, show_steps: bool = False):
		log = []
		a_is = []
		t_is = np.linalg.solve(basis.T, target_vector)
		for t in t_is:
			a_is.append(round(t))
		v = np.zeros(basis.shape[0])
		for i, a in enumerate(a_is):
			v_i = a * basis[i,:]
			v += v_i
		if show_steps == True:
			log.append(['','Basis', basis])
			log.append(['','Target Vector', target_vector])
			log.append(['', "Hadamard Ratio = ", Lattice.had_ratio(basis)])
			log.append(["1)", "t_is", t_is])
			log.append(["2)", "a_is", a_is])
			log.append(["3)", "Closest Vector", v])
			log.append(['', 'Distance ||w-v|| = ', np.linalg.norm(np.subtract(target_vector, v))])
			print(tabulate(log, tablefmt='grid'))
		return v


	
# ## Babai's algorithm for good basis
# good_basis = np.array([
# 	[-97, 19, 19], 
# 	[-18, 15, 43], 
# 	[-92, -32, 39]
# 	])
# target_vector = [53172, 81743, 3152]
# Lattice.babais_cvp(basis = good_basis, target_vector = target_vector, show_steps=True)

# ## Babai's algorithm for bad basis
# bad_basis = np.array([
# 	[-4179163, -1882253, 583183], 
# 	[-3184353, -1434201, 444361], 
# 	[-5277320, -2376852, 736426]
# 	])
# Lattice.babais_cvp(basis= bad_basis, target_vector = target_vector, show_steps=True)

		
