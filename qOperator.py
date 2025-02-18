# Here we write operators
########################################################################
import numpy as np
import pandas as pd
import qState as qS


def annihilationOp(n_qubits: int, pos: int) -> pd.DataFrame:
    simpleAnnihilation = np.array([[0, 1], [0, 0]])
    identity2 = np.array([[1,0],[0,1]])
    result = (pos == 1)*simpleAnnihilation + (pos != 1)*identity2
    for i in range(2, n_qubits + 1):
        if i == pos:
            result = np.kron(result, simpleAnnihilation)
        else:
            result = np.kron(result, identity2)
    states = qS.generateAllPossibleStates(n_qubits, [""])
    result = pd.DataFrame(result)
    result.index = states
    result.columns = states
    return pd.DataFrame(result)


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
    result = pd.DataFrame(result)
    result.index = states
    result.columns = states
    return pd.DataFrame(result)


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
                sumy += (creationOp(n_qubits, i).dot(annihilationOp(n_qubits, j)) +
                         annihilationOp(n_qubits, i).dot(creationOp(n_qubits, j)))
    return sumy
