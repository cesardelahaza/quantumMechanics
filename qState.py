import numpy as np

class QState:
    def __init__(self, arrayState):
        """
        Initialize a state
        :param arrayState: matrix array in this form [[constant, state]]. Example: if we want
        to write a|01> + b|10>, then we would do it like:
        [[a, [0,1]],[b,[1,0]]]
        """
        def showState(aState):
            l = len(aState)
            state = ''
            for i in range(l):
                constant_i = (aState[i])[0]
                if constant_i>0:
                    sign = '+'
                else:
                    sign = '-'
                state_i = (aState[i])[1]
                state = state + sign + ' ' + str(np.abs(constant_i)) + '|' + ''.join(map(str, state_i)) + '> '
            return state

        self.state = showState(arrayState)

        def binaryStateToDecimal(aState):
            string_state = ''.join(map(str, aState))
            return int(string_state, 2)
        def showStateDecimal(aState):
            l = len(aState)
            state = ''
            for i in range(l):
                constant_i = (aState[i])[0]
                if constant_i > 0:
                    sign = '+'
                else:
                    sign = '-'
                state_i = (aState[i])[1]
                state = state + sign + ' ' + str(np.abs(constant_i)) + '|' + str(binaryStateToDecimal(state_i)) + '> '
            return state
        self.decimal = showStateDecimal(arrayState)


#print(''.join((map(str, [0,0,1]))))
prim = QState([[2,[0,0,1]], [-2, [0,1,0]]])
print(prim.state)
print(prim.decimal)

#print('|'+ str(4) + ''.join(map(str, [0,0,1])) + '>')