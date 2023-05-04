import math
import random
import numpy as np
from utils import *
'''
Discrete Gaussian sampling as per 

Floating point arithmetic not implemented

'''

class DiscreteGaussian:

	def __init__(self,center:float, sigma:float, tao:float) -> None:
		'''
		samples points using discrete gaussian
		Parameters
		----------
		t: float
			center of the gaussian distribution
		sigma: float
			standard deviation of the gaussian distribution
		tao: float
			tailcut parameter, usually the sigma multiple distance from the center you want to cut
			P[-k\sigma < x < k\sigma] >= 1 - 2e^{-k^2/2}
		'''
		# self.m 		= m		#sample Z_m: Rejection sampling for discrete gaussian on Z
		self.t 		= center    
		self.sigma 	= sigma
		self.tao 	= tao


	def _rejectionSampling(self)-> int:
		'''
		creates an instance of rejection sampling
		'''
		h:float 	= - math.pi/(self.sigma**2)
		x_max		= math.ceil(self.t + self.tao * self.sigma)  
		x_min		= math.floor(self.t - self.tao * self.sigma)
		x:int		= random.randint(x_min, x_max)
		p:float		= math.exp(h * (x - self.t)**2)  ##fpm
		r:float		= random.random()  ##fpm
		if r <= p:
			return x

	def sample(self, n:int) -> list:
		'''
		using rejection sampling samples n vectors
		'''
		vec = list()
		self.counter = 0
		while len(vec) != n:
			x = self._rejectionSampling()
			self.counter += 1
			if x!= None:
				vec.append(x)
		return vec
	
class Uniform:	
	'''
	Uniform Distribution
	paramters
	----------
	a: int
		min
	b: int
		max 
	'''

	def __init__(self, low: int, high: int) -> None:
		self.x_min = low
		self.x_max = high

	def sample(self, n:int):
		'''
		samples 'n' numbers uniformly from the range (a,b)
		uses the built-in random function
		for Z_q use: a = 0, b = q
		'''
		vec = []
		for i in range(n):
			vec.append(np.random.randint(self.x_min, self.x_max))
		return vec
	

def discrete_gaussian_sampling(center, sigma, tao, sample_points):
	dis_gauss = DiscreteGaussian(center, sigma, tao)
	vec = dis_gauss.sample(sample_points)
	return vec

def uniform_sampling(a, b, sample_points):
	uni = Uniform(a, b)
	vec = uni.sample(sample_points)
	return vec
	
### what will be the expected number of runs before getting a sample point from rejection sampling	
# usually linear for same tao and sigma
# but what if sigma and tao varied??		 
	
# #=======================================
# d = DiscreteGaussian(center =  0, sigma=5, tao=3)
# print(d.sample(12))

# x = []
# y = []
# for i in range(0, 10000, 100):
# 	l = d.sample(i)
# 	y.append(d.counter)
# 	x.append(i)
# plt.plot(x, y)
# plt.show()


# uniform = Uniform(0, 256) ## sampling from Z_q where q = 256
# output = uniform.sample(10000)
# check_via_graph(output)


