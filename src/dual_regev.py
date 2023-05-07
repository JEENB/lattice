from lwe import *

class DualRegev:
    
	def __init__(self, q, n, alpha = None, r = None, m = None) -> None:
		self.q = q
		self.alpha = alpha
		self.r = r
		self.m = m
		self.n = n

		if r == None or m == None or alpha == None:
			m = 2 * n * math.ceil(math.log(q, math.e)) + 1
			r = math.ceil(math.log2(m))
			alpha = 1/(r * math.sqrt(m + 1)* math.log2(n))
			self.m, self.r, self.alpha = m, r, alpha
			print(f"** Parameters have been updated **\nm = {self.m},\nr = {self.r},\nalpha = {self.alpha}")

	def keygen(self):
		e = discrete_gaussian_sampling(0, self.r, 3, self.m)
		A, _ = LWE(LweParameter(self.q, self.n, self.alpha, discrete_gaussian_sampling, discrete_gaussian_sampling)).LWE_instances_matrix_version(self.m)
		b = np.dot(A, e) % self.q

		pk = (A, b)
		return e, pk
	
	def encrypt(self, message, pk):
		A = pk[0]
		b = pk[1]
		s = uniform_sampling(0, self.q, self.n)
		x = discrete_gaussian_sampling(0, self.alpha,3,  self.m)
		x_small = discrete_gaussian_sampling(0, self.alpha,3, self.m)[0]
		u = (np.dot(A.T, s) + x) % self.q
		v = (np.dot(b.T, s) + message * round(self.q/2) + x_small ) % self.q
		return (u, v)
	
	def decrypt(self, sk, ciphertext):
		uts = np.dot(np.array(sk).T, ciphertext[0])
		vminusuts = math.remainder((ciphertext[1]  - uts), self.q)
		if abs(vminusuts) < self.q / 4:
			return 0
		else:
			return 1
		
if __name__ == '__main__':	
	np.set_printoptions(threshold=5)
	##parameter setup	
	q = 25
	n = 10
	m = 10
	r = 10
	alpha = 10
	dreg = DualRegev(q, n,alpha, r, m)

	sk, pk = dreg.keygen() 
	message = [0,1,1,0,1,1,1,1,1,0]

	dec_message = []
	cipher = []
	for m in message:
		ciphertext = dreg.encrypt(m, pk)  # encrypting each message
		cipher.append(ciphertext)

		dec_message.append(dreg.decrypt(sk, ciphertext)) #decrypting each ciphertext
	print("List of Ciphertext: \n", tabulate(cipher, headers=['u', 'v'], tablefmt='pretty'))
	print("Decrypted Message: ", dec_message)