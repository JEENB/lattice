from grahm_schmidt import grahm_schmidt
import numpy as np
from lattice import *

def mu(b_i, b_j):
    if np.inner(b_j, b_j) != 0:
        return np.inner(b_i, b_j)/np.inner(b_j, b_j) 
    else:
        return 0

def LLL(basis: np.ndarray):
    n = basis.shape[0]

    k = 1
    ortho = grahm_schmidt(basis)
    while k < n:
        for j in range(k - 1, -1, -1):
            mu_kj = mu(basis[k ], ortho[j ])
            if abs(mu_kj) > 1/2:
                basis[k ] = basis[k ] - basis[j ] * round(mu_kj)
                ortho = grahm_schmidt(basis)

        inner_orth_kk = np.inner(ortho[k], ortho[k])
        mu_ = mu(basis[k ], ortho[k-1 ])**2
        inner_ortho_k_m1 = np.inner(ortho[k-1 ], ortho[k-1 ])

        if inner_orth_kk >= (0.75 - mu_) * inner_ortho_k_m1 :
            print("yes")
            k += 1
        else:
            basis[k ] , basis[k-1 ] = basis[k-1 ], basis[k ]
            ortho = grahm_schmidt(basis)
            k = max(k-1, 1)
        break
    return basis

if __name__ == '__main__':
    bs = np.array([[3,4,5], [6,7,8], [9,10,11]], dtype=np.longfloat)
    print(LLL(bs))       
    # print(Lattice.had_ratio(LLL(bs)))    
