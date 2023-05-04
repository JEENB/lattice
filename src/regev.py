'''
Implementation of the Regev's encryption scheme. 

'''

from lwe import *
from math import log2

class Regev:
    
	def __init__(self, lweParameter: LweParameter) -> None:
		self.lweParameter = lweParameter
		self.LWE = LWE(self.lweParameter)

	def keygen(self, m):
		assert m > self.lweParameter.n * log2(self.lweParameter.q), f"m has to be greater than {self.lweParameter.n * log2(self.lweParameter.q)} "
		self.m = m
		sk = self.LWE.get_secret()
		pk = self.LWE.LWE_instances_matrix_version(m)
		return sk, pk

	def encrypt(self, message: int, pk):
		assert message == 0 or message ==1, "Message has to be binary"
		r = uniform_sampling(0,2, self.m)
		u = np.matmul(pk[0], r) % self.lweParameter.q
		v = (np.dot(r, pk[1]) + message * round(self.lweParameter.q/2)) % self.lweParameter.q
		return u, v
	
	def decrypt(self, sk, ciphertext: tuple):
		uts = np.dot(ciphertext[0], sk)
		vminusuts = math.remainder((ciphertext[1]  - uts), self.lweParameter.q)
		if abs(vminusuts) < self.lweParameter.q / 4:
			return 0
		else:
			return 1
	
if __name__ == '__main__':	
	##parameter setup	
	par = LweParameter(2566, 40, 0.1, secret_sampling = uniform_sampling, error_sampling = discrete_gaussian_sampling)

	## inistiating the class
	reg = Regev(par)

	sk, pk = reg.keygen(455)  ## m needs to be greater than 32

	message = [0,1,1,0,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0]

	dec_message = []
	for m in message:
		ciphertext = reg.encrypt(m, pk)  # encrypting each message

		dec_message.append(reg.decrypt(sk, ciphertext)) #decrypting each ciphertext
	print("Decrypted Message: ", dec_message)