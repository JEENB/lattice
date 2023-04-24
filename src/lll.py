import numpy as np
from grahm_schmidt import *
from lattice import *
from typing import Tuple

class LLL:
    
	def __init__(self, lattice_object: Lattice ) -> None:
		self.basis = lattice_object.basis

	@staticmethod
	def swap(v1:np.ndarray, v2:np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
		v1, v2 = v2, v1
		return v1, v2

	
	def lll(self):
		k = 1
		n = self.basis.shape[0] ## dimension is the number of rows
		basis = self.basis
		mu, ortho = grahm_schmidt(n, basis)
		while k < n:
			for j in range(k-1,-1,-1):
				mu_kj = np.dot(basis[k,:], ortho[j,:])/np.linalg.norm(ortho[j,:])**2
				print(mu_kj)
				basis[k,:] = basis[k,:] - round(mu_kj)*basis[j,:]
				for l in range(j + 1):
					mu[k][l] = mu[k][l] - mu_kj * mu[j][l]
			
			if norm(ortho[k,:]) >= (0.75 - mu[k][k-1] ** 2) * norm(ortho[k-1,:]):
				k = k + 1
			else:
				basis[k-1, :], basis[k,:] = basis[k,:], basis[k-1, :]
				mu, ortho	= grahm_schmidt(n, basis)
				k = max(k-1, 1)
		return basis
	

l = Lattice(4,np.array([[105, 821, 404, 328], [881, 667, 644, 927], [181, 483, 87, 500], [893, 834, 732, 441]]))

ll = LLL(l)
print(Lattice.had_ratio(l.basis))
red_bas = ll.lll()
# print(Lattice.had_ratio(red_bas))
print(red_bas)
