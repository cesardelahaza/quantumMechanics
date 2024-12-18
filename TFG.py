import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open("prueba.txt") as f:
    lines = f.readlines()

lines = lines[1:] # we don't want the first line

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

# Now we construct the vector state (ket) and its hermitian (bra)
dict_state = {extractState(x, 8): [extractCoef(x, 8)] for x in lines}
conj_dict_state = {extractState(x, 8): [np.conj(extractCoef(x, 8))] for x in lines}
vector_state = pd.DataFrame(dict_state)
conj_vector_state = pd.DataFrame(conj_dict_state).transpose()

# Density matrix of state
density_matrix = conj_vector_state.dot(vector_state)

def density_matrix_1p(pos_qubit: int, arr: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced density matrix of the qubit in position pos_qubit
    :param pos_qubit: The qubit in which we are interested to calculate its density reduced matrix
    :param arr: The density matrix of the state
    :return: Reduced density matrix of the qubit
    """
    state_names = list(arr.columns)
    dim = len(state_names)
    sum00, sum01, sum10, sum11 = 0, 0, 0, 0
    for i in range(dim):
        for j in range(dim):
            row, col = state_names[i], state_names[j]
            rowEx, colEx = row[:pos_qubit-1] + row[pos_qubit:], col[:pos_qubit-1] + col[pos_qubit:]
            rq, cq = row[pos_qubit-1], col[pos_qubit-1]
            if rowEx == colEx:
                if rq == '0' and cq == '0':
                    sum00 += arr.loc[row][col]
                elif rq == '0' and cq == '1':
                    sum01 += arr.loc[row][col]
                elif rq == '1' and cq == '0':
                    sum10 += arr.loc[row][col]
                else:
                    sum11 += arr.loc[row][col]
    return pd.DataFrame(np.array([[sum00, sum01],[sum10, sum11]]), columns=['0','1'], index=['0','1'])

def density_matrix_2p(pos_qubit_1: int, pos_qubit_2: int, arr: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced density matrix of two qubits in positions pos_qubit_1 and pos_qubit_2
    :param pos_qubit_1: Position of first qubit
    :param pos_qubit_2: Position of second qubit
    :param arr: The density matrix of the state
    :return: Reduced density matrix of the two qubits
    """
    state_names = list(arr.columns)
    dim = len(state_names)
    sumy = pd.DataFrame(0, index=["00", "01", "10", "11"], columns=["00", "01", "10", "11"], dtype=complex)
    for i in range(dim):
        for j in range(dim):
            row, col = state_names[i], state_names[j]
            rowEx1, colEx1 = row[:pos_qubit_1 - 1] + row[pos_qubit_1:], col[:pos_qubit_1 - 1] + col[pos_qubit_1:]
            rowEx, colEx = rowEx1[:pos_qubit_2 - 1-1] + rowEx1[pos_qubit_2-1:], colEx1[:pos_qubit_2 - 1-1] + colEx1[pos_qubit_2-1:]
            rq1, cq1 = row[pos_qubit_1 - 1], col[pos_qubit_1 - 1]
            rq2, cq2 = row[pos_qubit_2 - 1], col[pos_qubit_2 - 1]
            rq, cq = rq1+rq2, cq1+cq2
            if rowEx == colEx:
                sumy.at[rq, cq] += arr.loc[row][col]
    return sumy

def eigenvals(arr: pd.DataFrame) -> list[float]:
    """
    Calculate the eigenvalues of the matrix arr
    :param arr: Matrix
    :return: List of eigenvalues of the matrix. In our case the eigenvalues are real, because our arr will be hermitian
    """
    eigs = np.linalg.eigvals(arr)
    eigs = list(map(lambda x: x.real, eigs))
    return eigs

def entropy(eigenvals_l: list) -> float:
    """
    Entropy of a set of eigenvalues
    :param eigenvals_l: List of eigenvalues
    :return: Entropy calculated with von Neumann entropy
    """
    return -sum([x*np.log(x) for x in eigenvals_l])

def EAM(arr: pd.DataFrame) -> pd.DataFrame:
    """
    Entanglement adjacency matrix
    :param arr: Density matrix
    :return: Entanglement adjacency matrix
    """
    dim = len(list(arr.columns)[0])
    eam = pd.DataFrame(0, index=np.arange(dim)+1, columns=np.arange(dim)+1, dtype=float)
    for i in range(dim):
        density_i = density_matrix_1p(i + 1, arr)
        s_i = entropy(eigenvals(density_i))
        # eam.at[i+1,i+1] = s_i/2 # diagonal
        for j in range(i+1, dim):
            density_j = density_matrix_1p(j+1, arr)
            s_j = entropy(eigenvals(density_j))
            density_i_j = density_matrix_2p(i+1, j+1, arr)
            s_i_j = entropy(eigenvals(density_i_j))
            eam.at[i+1,j+1] = (s_i+s_j-s_i_j)/2
    eam = eam + eam.transpose()
    return eam

### Results

#first_EAM = EAM(density_matrix)
#print(first_EAM)
eee = EAM(density_matrix)
plt.matshow(eee)
plt.show()
#densities = density_matrix_1p(1, density_matrix)
#print(densities)

#plt.matshow(first_EAM)
# Mirar colores aquí: https://matplotlib.org/stable/users/explain/colors/colormaps.html
#plt.show()
