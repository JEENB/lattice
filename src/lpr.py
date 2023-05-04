'''
Implementation of the Lyubashevsky, Peikert, Regev encryption scheme. 

'''

from lwe import *
from math import log2

class LPR:
    
	def __init__(self, lweParameter: LweParameter) -> None:
		assert lweParameter.secret_sampling == discrete_gaussian_sampling, "Secret sampling has to be discrete gaussian sampling"
		self.lweParameter = lweParameter
		self.LWE = LWE(self.lweParameter)

	def keygen(self, m):
		assert m == self.lweParameter.n * 2 + 1, f"Required m is {self.lweParameter.n * 2 + 1}"
		sk = self.LWE.get_secret()
		pk = self.LWE.LWE_instances_matrix_version(self.lweParameter.n)
		return sk, pk

	def encrypt(self, message: int, pk):
		assert message == 0 or message ==1, "Message has to be binary"
		r = discrete_gaussian_sampling(0, self.lweParameter.alpha * self.lweParameter.q, 3, self.lweParameter.n)
		x = discrete_gaussian_sampling(0, self.lweParameter.alpha * self.lweParameter.q, 3, self.lweParameter.n)
		xprime = discrete_gaussian_sampling(0, self.lweParameter.alpha * self.lweParameter.q, 3, 1)[0]
		u = (np.matmul(pk[0], r) + x) % self.lweParameter.q
		v = (np.dot(r, pk[1]) + xprime + message * round(self.lweParameter.q/2)) % self.lweParameter.q
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
	par = LweParameter(256, 30, 0.1, secret_sampling = discrete_gaussian_sampling, error_sampling = discrete_gaussian_sampling)

	## inistiating the class
	lpr = LPR(par)

	sk, pk = lpr.keygen(61) 

	message = [0,1,1,0,1,1,1,1,1,0]

	dec_message = []
	cipher = []
	for m in message:
		ciphertext = lpr.encrypt(m, pk)  # encrypting each message
		cipher.append(ciphertext)

		dec_message.append(lpr.decrypt(sk, ciphertext)) #decrypting each ciphertext
	print("List of Ciphertext: \n", tabulate(cipher, headers=['u', 'v'], tablefmt='pretty'))
	print("Decrypted Message: ", dec_message)