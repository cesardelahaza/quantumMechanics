import numpy as np
import pandas as pd

def hopping1Ham(n_qubits,p0: float, p1: float, periodic=False):
    mat = (np.diag(np.full(n_qubits,p0)) +
           np.diag(p1*np.ones(n_qubits-1),1) +
           np.diag(p1*np.ones(n_qubits-1),-1))
    if periodic:
        mat[n_qubits-1][0], mat[0][n_qubits-1] = p1, p1
    mat = pd.DataFrame(mat)
    mat.index = [i for i in range(1,n_qubits+1)]
    mat.columns = [i for i in range(1, n_qubits + 1)]
    return mat

def hoppingChain(n_places, n_qubits, pps):
    return None