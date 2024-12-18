import numpy as np
import pandas as pd
import createState

def kroneckerProduct(A: np.array, B: np.array) -> np.array:
    return pd.DataFrame(np.kron(A,B))
# The tensorial product is not associative in the sense of (AxB)xC=Ax(BxC),
# but it is in the sense that there is an isomorphism of the spaces (UxV)xW = Ux(VxW)

def annihilationOp(n_qubits: int, pos: int) -> pd.DataFrame:
    simpleAnnihilation = np.array([[0, 1], [0, 0]])
    identity2 = np.array([[1,0],[0,1]])
    result = (pos == 1)*simpleAnnihilation + (pos!=1)*identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = kroneckerProduct(result, simpleAnnihilation)
        else:
            result = kroneckerProduct(result, identity2)
    states = createState.generateAllPossibleStates(n_qubits, [""])
    result.index = states
    result.columns = states
    return result


def creationOp(n_qubits: int, pos: int) -> pd.DataFrame:
    simpleCreation = np.array([[0, 0], [1, 0]])
    identity2 = np.array([[1,0],[0,1]])
    result = (pos == 1)*simpleCreation + (pos!=1)*identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = kroneckerProduct(result, simpleCreation)
        else:
            result = kroneckerProduct(result, identity2)
    states = createState.generateAllPossibleStates(n_qubits, [""])
    result.index = states
    result.columns = states
    return result

def numOp(n_qubits: int, pos: int) -> pd.DataFrame:
    return creationOp(n_qubits, pos).dot(annihilationOp(n_qubits, pos))

def randomCoeffsH(n_qubits:int) -> pd.DataFrame:
    vectorProbs = np.random.dirichlet(np.ones(2**n_qubits * 2**n_qubits), size=1)[0]
    matrixCoeffs = np.reshape(vectorProbs, (2**n_qubits, 2**n_qubits))
    return pd.DataFrame(matrixCoeffs)

def randomH(n_qubits: int, ring: bool):
    states = createState.generateAllPossibleStates(n_qubits, [""])
    ham = pd.DataFrame(np.zeros((2**n_qubits, 2**n_qubits)))
    ham.index, ham.columns = states, states
    coeffs = randomCoeffsH(n_qubits)
    for i in range(n_qubits):
        for j in range(n_qubits):
            annihilate = annihilationOp(n_qubits, j+1)
            create = creationOp(n_qubits, i+1)
            ham -= coeffs[i][j] * create.dot(annihilate)
    # if ring:
    #     bottom_left, bottom_left[2**n_qubits-1,0] = np.zeros((2**n_qubits, 2**n_qubits)), 1
    #     top_right, top_right[0,2 ** n_qubits - 1] = np.zeros((2 ** n_qubits, 2 ** n_qubits)), 1
    #     ham -= coeffs[2**n_qubits-1][0]*bottom_left + coeffs[0][2**n_qubits-1]*top_right
    return ham

print(randomH(3, True))

def hamiltonian(arr: pd.DataFrame) -> pd.DataFrame:
    """
    Returns hamiltonian with the matrix representation of it
    :param arr:
    :return:
    """
    n_rows = len(arr.index)
    states = createState.generateAllPossibleStates(n_rows, [""])
    return None

def eigenH(hamiltonian: pd.DataFrame):
    return np.linalg.eig(hamiltonian)


# hhh = randomH(3, True)
# eee = eigenH(hhh)
# eigenVectors = eee.eigenvectors
# eigenValues = eee.eigenvalues
# eigenV1 = eigenVectors[1]



##############################
# Unidimensional Hopping Hamiltonian
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

def creationOpChain(n_qubits, ii):
    """
    This is de c+_i creation operator
    :param n_qubits:
    :param ii: respect to particle i in the chain
    :return:
    """
    return None


