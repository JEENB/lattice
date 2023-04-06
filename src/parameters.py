from math import sqrt





class NormalParameter:
	def __init__(self, center, sigma, tao ) -> None:
		self.t = center
		self.sigma = sigma
		self.tao = tao

class UniformParameter:
	def __init__(self, low, high) -> None:
		self.x_min = low
		self.x_max = high

class Parameters:
    
	def __init__(self, lwe: LweParameter, normal: NormalParameter, uniform: UniformParameter) -> None:

		self.q 		= lwe.q
		self.n 		= lwe.n
		self.alpha 	= lwe.alpha
		self.samples = lwe.samples
		self.error_sampling = lwe.error_sampling
		self.secret_sampling = lwe.secret_sampling

		self.t = normal.t
		self.sigma = normal.sigma
		self.tao = normal.tao

		self.x_min = uniform.x_min
		self.x_max = uniform.x_max