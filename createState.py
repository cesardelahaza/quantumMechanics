import itertools

import permutations

import numpy as np
# Create a function that makes random states that have nq qubits of |0> and |1>
def randomizeState(n_qubits: int, file_txt: str):
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

def generateAllPossibleStates(n_qubits: int, s: list[str]) -> list[str]:
    """

    :param n_qubits:
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

# Another to read the state

# Create ket

# Create bra

# Assign binary state to respective vector in the space
# For example |000> will be (1,0,0,0,0,0,0,0) and |010> will be (0,0,1,0,0,0,0,0)
# We can see this as the position of the number in binary form: 000 is 0, 010 is 2
def assignVectorToState(state:str) -> np.array:
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
    return v

def assignStateToVector(vector: np.array) -> str:
    """
    Inverse operation of assignVectorToState
    :param vector:
    :return:
    """
    for i in range(len(vector)):
        if vector[i]==1:
            break
    # NOT FINISHED!
    return None


#print(assignVectorToState("000"))

def generateNQubitsStates(n_qubits: int, how_many: int):
    ones = ['1' for i in range(how_many)]
    zeroes = ['0' for i in range(n_qubits-how_many)]
    return permutations(ones+zeroes)



def permutations(s):
    return sorted({"".join(p) for p in itertools.permutations(s)})

print(generateNQubitsStates(3, 1))