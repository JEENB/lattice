import math
import random
from decompose import Decompose
import numpy as np
import numpy.linalg as linalg

'''
Discrete Gaussian sampling as per 

Floating point arithmetic not implemented

'''

class DiscreteGaussian:

	def __init__(self, m:int, t:float, sigma:float, tao:float) -> None:
		'''
		
		'''
		self.m 		= m		#sample Z_m: Rejection sampling for discrete gaussian on Z
		self.t 		= t		    
		self.sigma 	= sigma
		self.tao 	= tao

	def _rejectionSampling(self)-> int:
		if self.t > self.m or self.sigma > self.m or self.tao > self.m:
			raise ValueError(f"Expected < {self.m}")
		h:float 	= - math.pi/(self.sigma**2)
		x_max:int 	= math.ceil(self.t + self.tao * self.sigma)  
		x_min:int 	= math.floor(self.t - self.tao * self.sigma)
		x:int		= random.randint(x_min, x_max)
		p:float		= math.exp(h * (x - self.t)**2)  ##fpm
		r:float		= random.random()  ##fpm
		return x, r, p

	def sample(self, n:int) -> list:
		vec = list()
		for _ in range(n):
			r = math.inf
			p = -math.inf
			while r >= p:
				x, r, p = self._rejectionSampling()
			vec.append(x)
		return vec
			 
	
#=======================================
d = DiscreteGaussian(m = 10,t =  0, sigma=5, tao=1)
l = d.sample(10)
# print(l.count(range(10)))


