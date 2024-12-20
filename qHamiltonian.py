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

# Let's define some probabilities of jump. Imagine we have the probability p0 of no jump,
# p1 the probability of jump with 1 particle, p2 the probability of jump with 2 particles

def numberOfParticles(s:str):
    return sum(list(map(int, s)))

def hoppingProbsMatrix(*probs):
    l = len(probs)
    states = createState.generateAllPossibleStates(l, [""])
    mat = pd.DataFrame(np.zeros((2 ** l, 2 ** l)))
    mat.index, mat.columns = states, states
    for i in range(2**l):
        for j in range(2**l):
            if i == j:
                mat.loc[states[i], states[j]] = probs[0]
            else:
                if numberOfParticles(states[i]) == numberOfParticles(states[j]):
                    mat.loc[states[i], states[j]] = probs[numberOfParticles(states[i])]
    return mat

print(hoppingProbsMatrix(0.5, 0.2, 0.3))

def hoppingHam(n_qubits, *probs):
    states = createState.generateAllPossibleStates(n_qubits, [""])
    ham = pd.DataFrame(np.zeros((2 ** n_qubits, 2 ** n_qubits)))
    ham.index, ham.columns = states, states
    return None
