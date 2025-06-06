# Here we define functions to help manage the states (strings)
########################################################################
import itertools
import pandas as pd
#import permutations
import numpy as np
from itertools import permutations as perm

# Create a function that makes random states that have nq qubits of |0> and |1>
def randomizeState(n_qubits: int, file_txt: str):
    """
    This function returns a random state
    :param n_qubits: number of qubits
    :param file_txt: name of file to write the state
    :return: random state
    """
    maxN = 2**n_qubits
    # We can create our numbers in this way:
    #   1. We assign a real probability p_s to each state in the global state.
    #   2. Then we multiply by exp(i*phi_s) where phi_s is a random phase [0,2\pi)
    #   3. Now we get our states multiplied by complex coefficients
    #################################################################
    # We generate de p_s with the Dirichlet distribution
    p2_s = np.random.dirichlet(np.ones(maxN), size=1)[0]
    p_s = [np.sqrt(p) for p in p2_s]
    # We generate random phases
    phi_s = [complex(np.exp(1j * x)) for x in np.random.uniform(0, 2 * np.pi, size=maxN)]
    # We create the coefficients
    coeffs = [complex(p_s[i] * phi_s[i]) for i in range(maxN)]
    # We create the states
    states = generateAllPossibleStates(n_qubits, [""])
    with open(file_txt, 'w') as output:
        output.writelines("# Wavefunction: \n")
        for i in range(maxN):
            real_coeff = coeffs[i].real
            complex_coeff = coeffs[i].imag
            output.writelines(states[i] + ": " + str(real_coeff) + " " + str(complex_coeff) + "\n")


def generateAllPossibleStates(n_qubits: int, s: list[str] = [""]) -> list[str]:
    """
    This function generates all possible states with n_qubits.
    Example: we have n_qubits=3, so this function will return
    [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0]...]
    :param n_qubits: number of qubits
    :param s: to initialize the function s=[""]
    :return:
    """
    if n_qubits == 0:
        return s
    else:
        s0 = [subs+"0" for subs in s]
        s1 = [subs+"1" for subs in s]
        return (generateAllPossibleStates(n_qubits-1, s0) +
                generateAllPossibleStates(n_qubits-1, s1))

# Assign binary state to respective vector in the space
# For example |000> will be (1,0,0,0,0,0,0,0) and |010> will be (0,0,1,0,0,0,0,0)
# We can see this as the position of the number in binary form: 000 is 0, 010 is 2


def assignStateToVector(state: str) -> pd.DataFrame:
    """
    Assign vector from basis in crescent order in relation to binary number (computational basis)
    For example: |00> is (1,0,0,0), |10> is (0,0,1,0)
    :param state: is in the form "010010" for example
    :return: decimal number of the state
    """
    l = len(state)
    sumy, count = 0, 0
    for s in state:
        count += 1
        sumy += int(s)*2**(l-count)
    v = np.zeros(2**l)
    v[sumy] = 1
    return pd.DataFrame(v)


def assignVectorToState(vector: pd.DataFrame) -> str:
    """
    Inverse operation of assignVectorToState
    :param vector: number of state
    :return: string denoting the state
    """
    lv = len(vector)
    for i in range(lv):
        if vector.iloc[i].item() != 0:
            break
    ls = int(np.log2(lv))
    s = ''
    for j in range(ls):
        s = str(i % 2) + s
        i = i//2
    return s


def permutations(s):
    return sorted({"".join(p) for p in itertools.permutations(s)})


def generateNQubitsStates(n_qubits: int, how_many: int):
    positions = itertools.combinations(range(n_qubits), how_many)
    return [''.join('1' if i in pos else '0' for i in range(n_qubits)) for pos in sorted(positions)][::-1]


def generateStates(n_qubits: int):
    states = []
    for i in range(n_qubits+1):
        s = generateNQubitsStates(n_qubits, int(i))
        states = states + s
    return states


def normalizeState(state: pd.DataFrame) -> pd.DataFrame:
    norm = np.sqrt(state.dot(state))
    return state/norm


def tuples(n_qubits, how_many):
    return list(perm(range(n_qubits), how_many))


def nStates(n_qubits, how_many):
    powers_of_2 = [2**i for i in range(n_qubits)]
    return sorted({sum(powers_of_2[i] for i in tup) for tup in tuples(n_qubits, how_many)})
