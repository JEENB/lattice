import numpy as np
from gram_schmid import gram_schmid
from lattice import *
from typing import Tuple

def norma(x):
	'''
	reutrns the norm squared of a vector x 
	||x||^2
	'''
	return np.power(math.sqrt(np.array([i ** 2 for i in x]).sum()),2)
# norma = lambda x: np.power(math.sqrt(np.array([i ** 2 for i in x]).sum()),2)

class LLL(Lattice):
    
	def __init__(self, dim: int, basis: np.ndarray) -> None:
		super().__init__(dim, basis)
		self.reduced_basis = None
	# def __init__(self, lattice_object: Lattice ) -> None:
	# 	self.lattice = lattice_object
	# 	self.basis = self.lattice.basis
	# 	self.reduced_basis = None

	def lll(self, delta=3/4):
		b = self.basis.copy()
		b_, m = gram_schmid(b)
		n = b.shape[1]
		k = 1
		norm_bstar = [np.inner(i,i) for i in b_]
		while k < n:
			for j in reversed(range(k)):
				mu= m[k][j]
				if abs(mu) > 0.5:
					q = round(mu)
					b[k] = np.subtract(b[k], np.multiply(q , b[j]))
					for l in range(j+1):
						m[k][l] = m[k][l] - q * m[j][l] 


			if norm_bstar[k] >= (delta - np.power(m[k][k - 1], 2)) * norm_bstar[k-1]:
				k += 1
			else:
				b[k], b[k - 1] = b[k - 1].copy(), b[k].copy()
				mup = m[k][k-1]
				B = norm_bstar[k] + (mup ** 2) * norm_bstar[k - 1]
				m[k][k-1] = mup * norm_bstar[k-1]/B
				norm_bstar[k] = norm_bstar[k] * norm_bstar[k-1]/B
				norm_bstar[k-1] = B
				for j in range(k-1):
					m[k-1][j], m[k][j] = m[k][j], m[k-1][j]
				for j in range(k+1, n):
					t = m[j][k]
					m[j][k] = m[j][k-1] - mup * t
					m[j][k-1] = t + m[k][k-1] * m[j][k]
				k = max(k-1, 1)
		self.reduced_basis = b
		return b
	
if __name__ == '__main__':

	basis = np.array([[3,4,5], [6,7,8], [9,10,11]])

	# basis = generate_lattice_points(15, 5000)
	
	#instantiating the class
	reduction = LLL(3, basis)

	# print("Basis: ", basis)
	print("Hadamard Ratio: ", Lattice.had_ratio(basis))
	
	red_bas= reduction.lll()
	print("Reduced Basis: ", red_bas)
	print("Hadamard Ratio: ", Lattice.had_ratio(red_bas))

	# plot_lattice_3D(basis[0], basis[1], basis[2], red_bas[0], red_bas[1], red_bas[2], points_count=5)
