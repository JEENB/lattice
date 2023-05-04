from lattice import *
import random

class GGH(Lattice):
	def __init__(self, dim, basis, delta = 3) -> None:
		Lattice.__init__(self, dim, basis)
		self.delta = delta

	def key_gen(self):
		print("** Key Generation **")
		self.sk = self.basis
		self.pk = self.generate_bad_basis(self.basis)
		print("Hadamard Ratio sk: ", self.hadamard_ratio)
		print("Hadamard Ratio pk: ", self.had_ratio(self.pk))
		
		return self.sk, self.pk

	def encrypt(self, message: list):
		'''
		TODO: error check
		'''
		print("\n** GGH Encryption **")
		assert len(message) == self.dim, "Dimension mismatch"
		r = [np.random.randint(-self.delta, self.delta) for _ in range(self.dim)]
		e = np.matmul(message, self.pk) + r
		return e
	
	def decrypt(self, ciphertext, sk):
		print("\n** GGH Decryption **")
		v = Lattice.babais_cvp(basis = sk, target_vector=ciphertext)
		m = np.dot(v,np.linalg.inv(self.pk))
		for i in range(len(m)):
			m[i] = round(m[i])
		return m
	
	def sign(self, message: list):
		print("\n** GGH Signing **")
		assert len(message) == self.dim, "Dimension mismatch"
		s = Lattice.babais_cvp(self.sk, message)
		a_is = np.linalg.solve(self.pk, s)
		return a_is
	
	def verify(self, signature: list, message):
		print("\n** GGH Verification **")
		s = np.dot(signature, self.pk.T)
		print(norm(np.subtract(s,message)))


if __name__ == '__main__':
	def pke():
		basis = np.array([[-97, 19, 19], [-18, 15, 43], [-92, -32, 39]])
		g = GGH(dim = 3, basis = basis )

		# key generation
		sk, pk = g.key_gen()
		print("Secret Key:\n", sk)
		print("Public Key:\n", pk)

		# encryption
		m = [678846, 651685, 160467]
		cipher = g.encrypt(m)
		print("CipherText: " , cipher)

		# decryption
		mes = g.decrypt(cipher, sk)
		print("DecryptedMessage: ",mes)

	def sig():
		basis = np.array([[-97, 19, 19], [-18, 15, 43], [-92, -32, 39]])
		g = GGH(dim = 3, basis = basis )

		# key generation
		sk, pk = g.key_gen()
		print("Secret Key:\n", sk)
		print("Public Key:\n", pk)

		#signing
		m = [678846, 651685, 160467]
		signature = g.sign(m)
		print("Signature: ", signature)

		#verification
		ver = g.verify(signature, m)
		print(ver)

		#message changed
		m_prime = [678846, 651685, 1604674]
		ver = g.verify(signature, m_prime)

	pke()
	sig()


# m_prime = [678846, 651685, 1604674]

# signature = g.sign(m)
# print(signature)

# ver = g.verify(signature, m_prime)
# print(ver)