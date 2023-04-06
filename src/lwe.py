from math import sqrt, log2, exp, pi
from sampling import *
from tabulate import tabulate

class LweParameter:
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
	'''
	def __init__(self, q:int, n:int, alpha:float, secret_sampling, error_sampling = discrete_gaussian_sampling) -> None:
		self.q 		= q
		self.n 		= n
		self.alpha 	= alpha
		try:
			self.alpha = alpha
			assert alpha * q == sqrt(n)
		except: 
			self.alpha = sqrt(n) /q
			print(f"New alpha is {self.alpha}")
		self.error_sampling = error_sampling
		self.secret_sampling = secret_sampling

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
	def __init__(self, param: LweParameter):
		self.param = param
		self.secret = self.__generate_secret()
		self.e = []

	def __generate_secret(self):
		'''
		Generates a secret of length n depending on the distribuiton. 
		Parameters:
		-------------
		distribution: Gaussian/ Uniform
			Default: Gaussian
		'''
		if self.param.secret_sampling == discrete_gaussian_sampling:
			return self.param.secret_sampling(center = 0, sigma = self.param.q * self.param.alpha, tao = 3, sample_points = self.param.n)  
		elif self.param.secret_sampling == uniform_sampling: 
			return self.param.secret_sampling(a = 0, b = self.param.q, sample_points = self.param.n)


	def an_instance(self):
		'''Generates one instance of LWE'''
		a_i = uniform_sampling(0, self.param.q, self.param.n)
		e_i = self.param.error_sampling(0, sigma= self.param.alpha*self.param.q, tao= 3, sample_points=1)
		self.e.append(e_i)
		b_i = np.mod(np.dot(a_i, self.secret) + e_i , self.param.q)
		return a_i, b_i


	def LWE_instances(self, m):
		'''Generates m LWE instances'''
		instances = []

		self.param.m = m
		for i in range(self.param.m):
			a_i, b_i = self.an_instance()
			instances.append((a_i, b_i))
			# print((i, a_i, b_i))
		print("Error is: ", self.e)
		return instances


	def LWE_instances_matrix_version(self, m):
		'''Matrix version of the LWE instances A, B'''
		A = np.zeros(shape = (self.param.n, m))
		b = []
		self.param.m = m
		for i in range(m):
			a_i, b_i = self.an_instance()
			A[:,i] = np.array(a_i)
			b.append(b_i)
		return A, b

	
	def exhaustive_search(self, param: LweParameter, instances, success_probability, summary: bool = False) -> list:

		'''
		returns a list of secret vectors that pass the test. 
		'''
		param = param
		guess_secret = []

		## t = log(n) > \sqrt(\omega(log n))
		t = math.log(param.n, 2)


		## m = (log(1 − \epsilon) − n log(2tαq + 1))/ log(2tα)
		m_required = (log2(success_probability) - param.n * log2(2*t*param.alpha*param.q + 1))/log2(2*t*param.alpha)

		# print("m_required = ", m_required)
		assert param.m >= m_required, f"Required {m_required} samples, got {param.m}"

		## defining the interval [-2.t.\alpha.q, 2.t.\alpha.q + 1] 
		lower_interval = -t*param.alpha*param.q
		upper_interval =  t*param.alpha*param.q + 1  
		# print("Interval for e_i is: ", (lower_interval, upper_interval))

		all_secret = generate_all_possible_sequence(param.q, param.n) ##generating all sequences of secret
		for j, s in enumerate(all_secret):
			counter = 0
			error = []
			for i in range(param.m):
				a_i = instances[i][0]
				b_i = instances[i][1]
				as_p = np.dot(a_i, s)
				e_i = as_p - b_i 
				e_i = math.remainder(e_i , param.q)
				if e_i >= lower_interval and e_i < upper_interval:
					error.append(e_i)
					counter += 1
				
			if counter == param.m:
				guess_secret.append(s)

		if summary == True:
			print("Secret is: ", self.secret)
			print("t = ", t)
			print("Interval for e_i is: ", (lower_interval, upper_interval))
			print("M required is", m_required)
			print("# Secret Space = ", j)
			print("# guess secret = ", len(guess_secret))

		return guess_secret

l = LweParameter(q = 13, n = 4, alpha = 0.1, error_sampling= discrete_gaussian_sampling, secret_sampling= discrete_gaussian_sampling)

lwe = LWE(l)
inst = lwe.LWE_instances(m = 19)
all_secret = lwe.exhaustive_search(param  = l, instances=inst, success_probability=0.95, summary=True)
print(all_secret)



		