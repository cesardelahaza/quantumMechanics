import pandas as pd
import numpy as np
import qFunctions as qF
import qDensity as qD
import scipy as sp
import qState as qS

def EAM(arr: pd.DataFrame) -> pd.DataFrame:
    """
    Entanglement adjacency matrix
    :rtype: object
    :param arr: Density matrix
    :return: Entanglement adjacency matrix
    """
    dim = len(list(arr.columns)[0])
    eam = pd.DataFrame(0, index=np.arange(dim) + 1, columns=np.arange(dim) + 1, dtype=float)

    for i in range(dim):
        density_i = qD.density_matrix_1p(i + 1, arr)
        s_i = qF.entropy(qF.eigenvals(density_i))

        for j in range(i + 1, dim):
            density_j = qD.density_matrix_1p(j + 1, arr)
            s_j = qF.entropy(qF.eigenvals(density_j))
            density_i_j = qD.density_matrix_2p(i + 1, j + 1, arr)
            s_i_j = qF.entropy(qF.eigenvals(density_i_j))
            eam.at[i + 1, j + 1] = (s_i + s_j - s_i_j) / 2
    eam = eam + eam.T
    return eam


def sparseEAM(arr, n_qubits, how_many):
    eam = sp.lil_matrix((n_qubits, n_qubits), dtype=float)
    state_names = qS.generateNQubitsStates(n_qubits, how_many)
    for i in range(n_qubits):
        density_i = qD.sparse_density_matrix_1p(i + 1, arr, state_names)
        eigens_i, _ = qF.sparseEigen(density_i, 2)
        s_i = qF.entropy(eigens_i)

        for j in range(i + 1, n_qubits):
            density_j = qD.sparse_density_matrix_1p(j + 1, arr, state_names)
            eigens_j, _ = qF.sparseEigen(density_j, 2)
            s_j = qF.entropy(eigens_j)
            density_i_j = qD.sparse_density_matrix_2p(i + 1, j + 1, arr, state_names)
            eigens_ij, _ = qF.sparseEigen(density_i_j, 4)
            s_i_j = qF.entropy(eigens_ij)
            eam.at[i + 1, j + 1] = (s_i + s_j - s_i_j) / 2
    eam = eam + eam.transpose()
    return eam
