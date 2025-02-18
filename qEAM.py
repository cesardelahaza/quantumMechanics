import pandas as pd
import numpy as np
import qFunctions as qF
import qDensity as qD


def EAM(arr: pd.DataFrame) -> pd.DataFrame:
    """
    Entanglement adjacency matrix
    :param arr: Density matrix
    :return: Entanglement adjacency matrix
    """
    dim = len(list(arr.columns)[0])
    eam = pd.DataFrame(0, index=np.arange(dim)+1, columns=np.arange(dim)+1, dtype=float)
    for i in range(dim):
        density_i = qD.density_matrix_1p(i + 1, arr)
        s_i = qF.entropy(qF.eigenvals(density_i))
        # eam.at[i+1,i+1] = s_i/2 # diagonal
        for j in range(i+1, dim):
            density_j = qD.density_matrix_1p(j+1, arr)
            s_j = qF.entropy(qF.eigenvals(density_j))
            density_i_j = qD.density_matrix_2p(i+1, j+1, arr)
            s_i_j = qF.entropy(qF.eigenvals(density_i_j))
            eam.at[i+1,j+1] = (s_i+s_j-s_i_j)/2
    eam = eam + eam.transpose()
    return eam

# Create plot for EAM
