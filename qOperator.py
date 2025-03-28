# Here we write operators
########################################################################
import numpy as np
import pandas as pd
import qState as qS
import scipy.sparse as sp


def annihilationOp(n_qubits: int, pos: int) -> pd.DataFrame:
    simpleAnnihilation = np.array([[0, 1], [0, 0]])
    identity2 = np.array([[1,0],[0,1]])
    result = (pos == 1)*simpleAnnihilation + (pos != 1)*identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = np.kron(result, simpleAnnihilation)
        else:
            result = np.kron(result, identity2)
    states = qS.generateAllPossibleStates(n_qubits)
    return pd.DataFrame(result, index=states, columns=states)


def creationOp(n_qubits: int, pos: int) -> pd.DataFrame:
    simpleCreation = np.array([[0, 0], [1, 0]])
    identity2 = np.array([[1, 0], [0, 1]])
    result = (pos == 1)*simpleCreation + (pos != 1)*identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = np.kron(result, simpleCreation)
        else:
            result = np.kron(result, identity2)
    states = qS.generateAllPossibleStates(n_qubits, [""])
    return pd.DataFrame(result, index=states, columns=states)


def num_i_Op(n_qubits: int, pos: int) -> pd.DataFrame:
    return creationOp(n_qubits, pos).dot(annihilationOp(n_qubits, pos))


def numberOp(n_qubits: int):
    sum = num_i_Op(n_qubits, 1)
    for i in range(2, n_qubits+1):
        sum += num_i_Op(n_qubits, i)
    return sum


def kinetic_operator(n_qubits, periodic: bool = True):
    sumy = np.array(np.zeros((2**n_qubits, 2**n_qubits)))
    for i in range(1, n_qubits):
        sumy += (creationOp(n_qubits, i).dot(annihilationOp(n_qubits, i+1)) +
                 annihilationOp(n_qubits, i).dot(creationOp(n_qubits, i+1)))
    if periodic:
        sumy += (creationOp(n_qubits,n_qubits).dot(annihilationOp(n_qubits, 1)) +
                 annihilationOp(n_qubits, n_qubits).dot(creationOp(n_qubits, 1)))
    return sumy


def chemical_potential_operator(n_qubits):
    sumy = np.array(np.zeros((2**n_qubits, 2**n_qubits)))
    for i in range(n_qubits):
        sumy += num_i_Op(n_qubits, i+1)
    return sumy


def jump_op(n_qubits):
    sumy = np.array(np.zeros((2**n_qubits, 2**n_qubits)))
    for i in range(1, n_qubits-1):
        sumy += (creationOp(n_qubits, i).dot(annihilationOp(n_qubits, i+2)) +
                 annihilationOp(n_qubits, i).dot(creationOp(n_qubits, i+2)))
    return sumy


def connect_op(n_qubits, ls1, ls2):
    sumy = np.array(np.zeros((2 ** n_qubits, 2 ** n_qubits)))
    for i in ls1:
        for j in ls2:
            if i != j:
                sumy += (creationOp(n_qubits, i) @ annihilationOp(n_qubits, j) +
                         annihilationOp(n_qubits, i) @ creationOp(n_qubits, j))
    return sumy


def sparseAnnihilationOp(n_qubits: int, pos: int):
    simpleAnnihilation = sp.csr_matrix([[0, 1], [0, 0]])
    identity2 = sp.eye(2, format="csr")
    result = simpleAnnihilation if pos == 1 else identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = sp.kron(result, simpleAnnihilation, format="csr")
        else:
            result = sp.kron(result, identity2, format="csr")
    return result


def sparseCreationOp(n_qubits: int, pos: int):
    simpleCreation = sp.csr_matrix([[0, 0], [1, 0]])
    identity2 = sp.eye(2, format="csr")
    result = simpleCreation if pos == 1 else identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = sp.kron(result, simpleCreation, format="csr")
        else:
            result = sp.kron(result, identity2, format="csr")
    return result


def sparseConnectOp(n_qubits, ls1, ls2):
    sumy = sp.csr_matrix((2 ** n_qubits, 2 ** n_qubits))
    for i in ls1:
        for j in ls2:
            if i != j:
                sumy += (sparseCreationOp(n_qubits, i) @ sparseAnnihilationOp(n_qubits, j) +
                         sparseAnnihilationOp(n_qubits, i) @ sparseCreationOp(n_qubits, j))
    return sumy

