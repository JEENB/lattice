from lattice import *
import random

class GGH(Lattice):
	def __init__(self, dim, basis) -> None:
		Lattice.__init__(self, dim, basis)
		self.delta = 5

	def key_gen(self):
		self.sk = self.basis
		self.pk = self.generate_bad_basis()
		return self.sk, self.pk

	def encrypt(self, pk, message):
		'''
		TODO: error check
		'''
		# r = [random.choice(range(-self.delta, self.delta), self.dim)]
		r = [-4,-3,2]
		e = np.matmul(message, pk) + r
		return e

# l = Lattice(dim=3, basis=np.array([
# 									[-97, 19, 19], 
# 									[-36, 30, 86],
# 									[-184, -64, 78]
# 									]))
l = GGH(dim=3, basis=np.array([
									[-97, 19, 19], 
									[-36, 30, 86],
									[-184, -64, 78]
									]))
sk, pk = l.key_gen()
print(pk)

print(l.encrypt(pk, [86,-35, -32]))