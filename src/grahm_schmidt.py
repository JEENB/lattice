import numpy as np

class DependentBasis(Exception):
    '''Basis are not linearly independent'''

## represent basis as row vectors
def grahm_schmidt(dim, basis):
    
    if not isinstance(basis, np.ndarray):
        raise TypeError("Expected Numpy array")

    if type(basis) != float:
        basis = basis.astype(float)
        

    if np.linalg.matrix_rank(basis) != dim:
        raise DependentBasis("Given basis vectors are not linearly independent")

    mu = np.identity(dim)

    for i, vec in enumerate(basis):
        #print(i, vec)
        s = np.zeros(dim, dtype=float)
        for j in range(i):
            #print(vec, basis[j])
            #print("dot =", np.dot(vec,basis[j]))
            #print("norm =", np.linalg.norm(basis[j])**2)
            m = np.dot(vec, basis[j])/np.linalg.norm(basis[j])**2
            mu[i,j] = m
            #print("m =",m)
            s += m*basis[j]
            #print("s = ", s)
        basis[i] = np.subtract(basis[i], s)
        #print(basis[i])
        #print("---------------------")
    # print(mu[2])    
    return mu, basis

# print(grahm_schmidt(basis = np.array([[0,1,2], [1,4,0], [3,0,4]]), dim=3))
