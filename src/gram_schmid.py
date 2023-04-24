import numpy as np
class DependentBasis(Exception):
    '''Basis are not linearly independent'''

## represent basis as row vectors
def gram_schmid(basis: np.ndarray):
    
    if type(basis) != float:
        basis = basis.astype(float)
        
    dim = basis.shape[1]
    b_star = np.zeros(basis.shape)
    mu = np.zeros(basis.shape)
    b_star[0, :] = basis[0, :]
    for i in range(1, dim):
        v = basis[i, :]
        for j in range(i-1, -1, -1):
            m = np.dot(basis[i,:], b_star[j,:])/np.dot(b_star[j,:], b_star[j,:]) 
            mu[i,j] = m
            v = v -  m * b_star[j,:]
        b_star[i,:] = v 
    return b_star.astype('int'), mu

if __name__ == '__main__':
    basis = np.array([[1,1,1,], [-1,0,2], [3,5,6]])
    bas1 = gram_schmid(basis)
    print(bas1)
