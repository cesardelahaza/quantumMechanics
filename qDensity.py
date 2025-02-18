import qFunctions as qF
import pandas as pd
import numpy as np
import qState as cs


def density_matrix(states, nQ):
    dict_state = {qF.extractState(x, nQ): [qF.extractCoef(x, nQ)] for x in states}
    conj_dict_state = {qF.extractState(x, nQ): [np.conj(qF.extractCoef(x, nQ))] for x in states}
    vector_state = pd.DataFrame(dict_state)
    conj_vector_state = pd.DataFrame(conj_dict_state).transpose()
    return conj_vector_state.dot(vector_state)


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


def reduced_density_matrix(positions: list[int], arr):
    """
    Calculate the reduced density matrix of a density matrix
    :param positions: in ascending order
    :param arr: density matrix
    :return: reduced density matrix of the qubits in positions
    """
    all_states = list(arr.columns)
    dim = len(all_states[0]) # how many qubits are there
    qubits = len(positions) # how many qubits we select
    states = cs.generateAllPossibleStates(qubits, [""])
    sumy = pd.DataFrame(0, index=states, columns=states, dtype=float)
    rest_states = cs.generateAllPossibleStates(dim-qubits, [""])
    for i in states:
        for j in states:
            for el in range(len(rest_states)):
                ix = list(map(lambda x: qF.insert_qubits_list(positions, i, x), rest_states))
                jx = list(map(lambda x: qF.insert_qubits_list(positions, j, x), rest_states))
                sumy.at[i,j] += arr.loc[ix[el]][jx[el]]
    return sumy


def reduced_density_matrix_chain(positions: list[int], arr):
    all_states = list(arr.columns) # Example: 10000 01000 00100 00010 00001
    qubits = len(positions)  # how many qubits we select
    states = cs.generateAllPossibleStates(qubits, [""])
    sumy = pd.DataFrame(0, index=states, columns=states, dtype=float)
    rest_states = list(set((map(lambda x: qF.delete_qubits_list(positions, x), all_states))))
    for i in states:
        for j in states:
            for el in range(len(rest_states)):
                ir = list(map(lambda x: qF.insert_qubits_list(positions, i, x), rest_states))
                jr = list(map(lambda x: qF.insert_qubits_list(positions, j, x), rest_states))
                if ir[el] in all_states and jr[el] in all_states:
                    sumy.at[i, j] += arr.loc[ir[el]][jr[el]]
    return sumy
