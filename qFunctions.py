# SOME COMMON FUNCTIONS TO CALCULATE and MANIPULATE
########################################################################
import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import eigsh


def extractState(s: str, nQ: int) -> str:
    """
    Of each line of the .txt it returns the state. For example, it returns 01010101
    :param s: String with state and coefficient
    :param nQ: Number of qubits (in our case)
    :return: The state of the string state
    """
    state = s[:nQ]
    return state


def extractCoef(s: str, nQ: int) -> complex:
    """
    Of each line of the .txt it returns the coefficient as a complex number
    :param s: String with state and coefficient
    :param nQ: Number of qubits (in our case)
    :return: The coefficient of the string state
    """
    coefState = (s[nQ + 1:]).split()
    coef = float(coefState[0]) + float(coefState[1])*1j
    return coef


def eigenvals(arr: pd.DataFrame) -> list[float]:
    """
    Calculate the eigenvalues of the matrix arr
    :param arr: Matrix
    :return: List of eigenvalues of the matrix. In our case the eigenvalues are real, because our arr will be hermitian
    """
    eigs = np.linalg.eigvals(arr)
    eigs = list(map(lambda x: x.real, eigs))
    return eigs


def eigenstates(arr: pd.DataFrame):
    return np.linalg.eig(arr).eigenvectors.transpose()


def entropy(eigenvals_l: list) -> float:
    """
    Entropy of a set of eigenvalues
    :param eigenvals_l: List of eigenvalues
    :return: Entropy calculated with von Neumann entropy
    """
    return -sum([sp.special.xlogy(x,x) for x in eigenvals_l if x>0])


def insert_qubits_list(positions: list[int], qubits_string, rest_string):
    """
    Inserting qubits abc... with respective positions ijk... into rest_string. For example:
    111 into positions 1, 2 and 4 (of the result) with respect to 000: 110100.
    In total there are 6 qubits
    :param positions:
    :param qubits_string:
    :param rest_string:
    :return:
    """
    n = 0
    result = rest_string
    for i in qubits_string:
        pos = positions[n]
        result = result[:pos-1] + i + result[pos-1:]
        n += 1
    return result


def delete_qubits_list(positions: list[int], string):
    """
    Delete positions of positions list of the string
    :param positions: it is in ascending order
    :param string:
    :return:
    """
    pos = positions[::-1]
    result = string
    for i in pos:
        result = result[:i-1] + result[i:]
    return result


def sparseEigen(m, k=1):
    """
    Selects the first k lowest eigenvalues and their eigenvectors
    :param m: the matrix
    :param k: how many eigenvals we want
    :return:
    """
    return eigsh(m, k=k, which='SA')