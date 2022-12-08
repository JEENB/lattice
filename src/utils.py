''' linear algebra utils'''
import numpy.linalg as linalg
import numpy as np
import random

MIN = -99999999999999
MAX = 999999999999999

def dot(vec1:list, vec2:list) -> int:
	'''
	Computes the dot product between two vectors. 
	np.dot may give errors
	'''
	if len(vec1) != len(vec2):
		raise ValueError(f"Vec1 and Vec2 are of different length.")
	s = 0
	for i in range(len(vec1)):
		s += vec1[i]*vec2[i]
	return s
	
def norm(vector:list):
	'''
	returns the eucledian norm of the vector
	'''
	return linalg.norm(vector)


# def rand_unimod(n):
#     l = np.tril(np.array(random.sample(range(MIN, MAX), n*n)).reshape(n,n)).astype('float')
#     u = np.triu(np.array(random.sample(range(MIN, MAX), n*n)).reshape(n,n)).astype('float')
#     for i in range(0, n):
#         l[i, i] = u[i, i] = 1.0
#         if i < n - 1:
#             val = sum([l[i, j] * u[j, n-1] for j in range(0, i)])
#             u[i, n-1] = (1 - val) / l[i, i]
#         else:
#             val = sum([l[i, j] * u[j, n-1] for j in range(1, i+1)])
#             l[n-1, 0] = (1 - val) / u[0, n-1]
#     return dot(l, u)

# print(rand_unimod(3))