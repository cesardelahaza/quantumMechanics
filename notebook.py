# Here we write some results with the help of control.py
########################################################################
########################################################################
# IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
###############################
import control as ctl
import qDensity as qD
import qEAM
import qState as qS
import qOperator as qO
import qFunctions as qF
import plot
import networkx as nx
########################################################################
# MAIN: control
########################################################################

# Extract geometry of states
if ctl.wavefunction_1 == 1:
    with open("wavefunction_1.txt") as f:
        states = f.readlines()
        states = states[1:]  # we don't want the first line
    densityM = qD.density_matrix(states, 8)
    eam = qEAM.EAM(densityM)
    plt.matshow(eam, cmap='Blues')
    plt.show()

if ctl.wavefunction_2 == 1:
    with open("wavefunction_2.txt") as f:
        states = f.readlines()
        states = states[1:]  # we don't want the first line
    densityM = qD.density_matrix(states, 8)
    eam = qEAM.EAM(densityM)
    plt.matshow(eam, cmap='Blues')
    plt.show()

# #########################################################################
# Half-filling cases in chain of n places and m=n/2 occupied places
# Half-filling with n=4 and m=2
if ctl.hf_4q == 1:
    hf_states = qS.generateNQubitsStates(4, 2)

    mat_ham = (-qO.kinetic_operator(4, False).loc[hf_states, hf_states] -
               qO.chemical_potential_operator(4).loc[hf_states, hf_states])

    vecs = qF.eigenstates(mat_ham)
    vals = qF.eigenvals(mat_ham)
    vector_0 = vecs[0]
    density = pd.DataFrame(np.outer(vector_0, vector_0), index=hf_states, columns=hf_states)

    d1 = qD.reduced_density_matrix_chain([1], density)
    print("Entropy of d1 is " + str(qF.entropy(qF.eigenvals(d1))))
    d2 = qD.reduced_density_matrix_chain([1, 2], density)
    print("Entropy of d2 is " + str(qF.entropy(qF.eigenvals(d2))))
    d3 = qD.reduced_density_matrix_chain([1, 2, 3], density)
    print("Entropy of d3 is " + str(qF.entropy(qF.eigenvals(d3))))
    d4 = qD.reduced_density_matrix_chain([1, 2, 3, 4], density)
    print("Entropy of d4 is " + str(qF.entropy(qF.eigenvals(d4))))

# #########################################################################
# Half-filling with n=6 and m=3
if ctl.hf_6q == 1:
    hf_states = qS.generateNQubitsStates(6, 3)

    mat_ham = (-qO.kinetic_operator(6, False).loc[hf_states, hf_states] -
               qO.chemical_potential_operator(6).loc[hf_states, hf_states])

    vecs = qF.eigenstates(mat_ham)
    vals = qF.eigenvals(mat_ham)
    vector_0 = vecs[0]
    density = pd.DataFrame(np.outer(vector_0, vector_0), index=hf_states, columns=hf_states)

    d1 = qD.reduced_density_matrix_chain([1], density)
    print("Entropy of d1 is " + str(qF.entropy(qF.eigenvals(d1))))
    d2 = qD.reduced_density_matrix_chain([1, 2], density)
    print("Entropy of d2 is " + str(qF.entropy(qF.eigenvals(d2))))
    d3 = qD.reduced_density_matrix_chain([1, 2, 3], density)
    print("Entropy of d3 is " + str(qF.entropy(qF.eigenvals(d3))))
    d4 = qD.reduced_density_matrix_chain([1, 2, 3, 4], density)
    print("Entropy of d4 is " + str(qF.entropy(qF.eigenvals(d4))))
    d5 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5], density)
    print("Entropy of d5 is " + str(qF.entropy(qF.eigenvals(d5))))
    d6 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5, 6], density)
    print("Entropy of d6 is " + str(qF.entropy(qF.eigenvals(d6))))

# #########################################################################
# Half-filling with n=8 and m=4
if ctl.hf_8q == 1:
    hf_states = qS.generateNQubitsStates(8, 4)

    mat_ham = (-qO.kinetic_operator(8, False).loc[hf_states, hf_states] -
               qO.chemical_potential_operator(8).loc[hf_states, hf_states])

    vecs = qF.eigenstates(mat_ham)
    vals = qF.eigenvals(mat_ham)
    vector_0 = vecs[1]
    density = pd.DataFrame(np.outer(vector_0, vector_0), index=hf_states, columns=hf_states)

    d1 = qD.reduced_density_matrix_chain([1], density)
    print("Entropy of o1 is " + str(qF.entropy(qF.eigenvals(d1))))
    d2 = qD.reduced_density_matrix_chain([1, 2], density)
    print("Entropy of o2 is " + str(qF.entropy(qF.eigenvals(d2))))
    d3 = qD.reduced_density_matrix_chain([1, 2, 3], density)
    print("Entropy of o3 is " + str(qF.entropy(qF.eigenvals(d3))))
    d4 = qD.reduced_density_matrix_chain([1, 2, 3, 4], density)
    print("Entropy of o4 is " + str(qF.entropy(qF.eigenvals(d4))))
    d5 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5], density)
    print("Entropy of o5 is " + str(qF.entropy(qF.eigenvals(d5))))
    d6 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5, 6], density)
    print("Entropy of o6 is " + str(qF.entropy(qF.eigenvals(d6))))
    d7 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5, 6, 7], density)
    print("Entropy of o7 is " + str(qF.entropy(qF.eigenvals(d7))))
    d8 = qD.reduced_density_matrix_chain([1, 2, 3, 4, 5, 6, 7, 8], density)
    print("Entropy of o8 is " + str(qF.entropy(qF.eigenvals(d8))))

# #########################################################################
# Geometries and EAM
# 8 qubits non-periodic chain with half-filling
if ctl.geom_hf_8q_noperiodic == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    mat_ham = -qO.kinetic_operator(8, False).loc[hf_states, hf_states]
    mat_con = qO.kinetic_operator(8, False).loc[one_states, one_states]

    # plot.eam_plot(mat_con)
    plot.adjacency_matrix_graph(mat_con, 'circular')

    hf_values = qF.eigenvals(mat_ham)
    hf_min_value = np.argmin(hf_values)
    hf_vectors = qF.eigenstates(mat_ham)
    hf_fs = hf_vectors[hf_min_value]

    hf_density = pd.DataFrame(np.outer(hf_fs, hf_fs), index=hf_states, columns=hf_states)
    hf_eam = qEAM.EAM(hf_density)

    # plot.eam_plot(hf_eam, True)

    plot.compare_plot(mat_con, hf_eam, "8 qubits - Half-filling chain - Non-Periodic")

# #########################################################################
# 8 qubits periodic chain with half-filling
if ctl.geom_hf_8q_periodic == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    mat_ham = -qO.kinetic_operator(8, True).loc[hf_states, hf_states]
    mat_con = qO.kinetic_operator(8, True).loc[one_states, one_states]

    # plot.eam_plot(mat_con)
    plot.adjacency_matrix_graph(mat_con, 'circular')

    hf_values = qF.eigenvals(mat_ham)
    hf_min_value = np.argmin(hf_values)
    hf_vectors = qF.eigenstates(mat_ham)
    hf_fs = hf_vectors[hf_min_value]

    hf_density = pd.DataFrame(np.outer(hf_fs, hf_fs), index=hf_states, columns=hf_states)
    hf_eam = qEAM.EAM(hf_density)

    # plot.eam_plot(hf_eam, True)

    plot.compare_plot(mat_con, hf_eam, "8 qubits - Half-filling chain - Periodic")

# #########################################################################
# Connecting qubits with a qubit between: 1 with 3, 2 with 4, 3 with 5... (Half-filling)
if ctl.geom_hf_8q_jumps == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    mat_ham = -qO.jump_op(8).loc[hf_states, hf_states]
    mat_con = qO.jump_op(8).loc[one_states, one_states]

    # plot.eam_plot(mat_con)
    plot.adjacency_matrix_graph(mat_con, 'circular')

    hf_values = qF.eigenvals(mat_ham)
    hf_min_value = np.argmin(hf_values)
    hf_vectors = qF.eigenstates(mat_ham)
    hf_fs = hf_vectors[hf_min_value]

    hf_density = pd.DataFrame(np.outer(hf_fs, hf_fs), index=hf_states, columns=hf_states)
    hf_eam = qEAM.EAM(hf_density)

    # plot.eam_plot(hf_eam)

    plot.compare_plot(mat_con, hf_eam, "8 qubits - Connecting qubits separated with a position between them")

# #########################################################################
# Connecting qubits with a qubit between: 1 with 3, 2 with 4, 3 with 5... (Half-filling)
if ctl.geom_hf_8q_jumps2 == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    mat_jump = qO.connect_op(8,[1], [3,7]) + qO.connect_op(8, [5], [3,7]) +\
        qO.connect_op(8, [2],[4,8]) + qO.connect_op(8, [6], [4,8])
    mat_ham = -mat_jump.loc[hf_states, hf_states]
    mat_con = mat_jump.loc[one_states, one_states]

    # plot.eam_plot(mat_con)
    plot.adjacency_matrix_graph(mat_con, 'circular')

    hf_values = qF.eigenvals(mat_ham)
    hf_min_value = np.argmin(hf_values)
    hf_vectors = qF.eigenstates(mat_ham)
    hf_fs = hf_vectors[hf_min_value]

    hf_density = pd.DataFrame(np.outer(hf_fs, hf_fs), index=hf_states, columns=hf_states)
    hf_eam = qEAM.EAM(hf_density)

    # plot.eam_plot(hf_eam)

    plot.compare_plot(mat_con, hf_eam, "8 qubits - Connecting qubits separated with a position between them - Periodic")

# #########################################################################
# Connect qubits 1,2,3,4 between them and 5,6,7,8 between them, and 4 and 5 are connected (Half-filling)
if ctl.geom_hf_8q_blocks == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    two_matrix = (-qO.connect_op(8, [1, 2, 3, 4], [1, 2, 3, 4]).loc[hf_states, hf_states] -
                  qO.connect_op(8, [5, 6, 7, 8], [5, 6, 7, 8]).loc[hf_states, hf_states] -
                  qO.connect_op(8, [4, 5], [4, 5]).loc[hf_states, hf_states])
    con_matrix = (qO.connect_op(8, [1, 2, 3, 4], [1, 2, 3, 4]).loc[one_states, one_states] +
                  qO.connect_op(8, [5, 6, 7, 8], [5, 6, 7, 8]).loc[one_states, one_states] +
                  qO.connect_op(8, [4, 5], [4, 5]).loc[one_states, one_states])

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'circular')

    two_values = qF.eigenvals(two_matrix)
    two_min_value = np.argmin(two_values)
    two_vectors = qF.eigenstates(two_matrix)
    two_fs = two_vectors[two_min_value]

    two_density = pd.DataFrame(np.outer(two_fs, two_fs), index=hf_states, columns=hf_states)
    two_eam = qEAM.EAM(two_density)

    # plot.eam_plot(two_eam)

    plot.compare_plot(con_matrix, two_eam, "Blocks")

# #########################################################################
# This is [1,2,3]-[4]-[5]-[6,7,8] (Half-filling)
if ctl.geom_hf_8q_bridge == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    bridge_matrix = (-qO.connect_op(8, [4], [1, 2, 3]).loc[hf_states, hf_states] -
                     qO.connect_op(8, [5], [6, 7, 8]).loc[hf_states, hf_states] -
                     qO.connect_op(8, [4], [5]).loc[hf_states, hf_states])
    con_matrix = (qO.connect_op(8, [4], [1, 2, 3]).loc[one_states, one_states] +
                  qO.connect_op(8, [5], [6, 7, 8]).loc[one_states, one_states] +
                  qO.connect_op(8, [4], [5]).loc[one_states, one_states])

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'circular')

    bridge_values = qF.eigenvals(bridge_matrix)
    bridge_min_value = np.argmin(bridge_values)
    bridge_vectors = qF.eigenstates(bridge_matrix)
    bridge_fs = bridge_vectors[bridge_min_value]

    bridge_density = pd.DataFrame(np.outer(bridge_fs, bridge_fs), index=hf_states, columns=hf_states)
    bridge_eam = qEAM.EAM(bridge_density)

    # plot.eam_plot(bridge_eam)

    plot.compare_plot(con_matrix, bridge_eam, "8 qubits in bridge geometry")
# #########################################################################
# This is [1,2,3,4]-[5]-[6]-[7,8,9,10] (Half-filling)
if ctl.geom_hf_10q_bridge == 1:
    t_0 = time.perf_counter()

    one_states = qS.generateNQubitsStates(10, 1)[::-1]
    hf_states = qS.generateNQubitsStates(10, 5)

    bridge_matrix = (-qO.connect_op(10, [5], [1, 2, 3, 4]).loc[hf_states, hf_states] -
                     qO.connect_op(10, [6], [7, 8, 9, 10]).loc[hf_states, hf_states] -
                     qO.connect_op(10, [5], [6]).loc[hf_states, hf_states])
    con_matrix = (qO.connect_op(10, [5], [1, 2, 3, 4]).loc[one_states, one_states] +
                  qO.connect_op(10, [6], [7, 8, 9, 10]).loc[one_states, one_states] +
                  qO.connect_op(10, [5], [6]).loc[one_states, one_states])

    t_f = time.perf_counter()
    print("Time creating matrix: " + str(t_f - t_0))

    # plot.eam_plot(con_matrix)
    # t_fig1 = time.perf_counter()
    # print("Time creating fig1: " + str(t_fig1 - t_f))

    bridge_values = qF.eigenvals(bridge_matrix)
    bridge_min_value = np.argmin(bridge_values)
    bridge_vectors = qF.eigenstates(bridge_matrix)
    bridge_fs = bridge_vectors[bridge_min_value]

    bridge_density = pd.DataFrame(np.outer(bridge_fs, bridge_fs), index=hf_states, columns=hf_states)
    bridge_eam = qEAM.EAM(bridge_density)

    # plot.eam_plot(bridge_eam)
    # t_fig2 = time.perf_counter()
    # print("Time creating fig2: " + str(t_fig2 - t_fig1))

    plot.compare_plot(con_matrix, bridge_eam, "10 qubits in bridge geometry")

# #########################################################################
# Rainbow (Half-filling)
if ctl.geom_hf_8q_rainbow == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    rainbow_matrix = -(qO.connect_op(8, [1], [8]).loc[hf_states, hf_states] +
                       qO.connect_op(8, [2], [7]).loc[hf_states, hf_states] +
                       qO.connect_op(8, [3], [6]).loc[hf_states, hf_states] +
                       qO.connect_op(8, [4], [5]).loc[hf_states, hf_states])
    con_matrix = (qO.connect_op(8, [1], [8]).loc[one_states, one_states] +
                  qO.connect_op(8, [2], [7]).loc[one_states, one_states] +
                  qO.connect_op(8, [3], [6]).loc[one_states, one_states] +
                  qO.connect_op(8, [4], [5]).loc[one_states, one_states])

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    rainbow_values = qF.eigenvals(rainbow_matrix)
    rainbow_min_value = np.argmin(rainbow_values)
    rainbow_vectors = qF.eigenstates(rainbow_matrix)
    rainbow_fs = rainbow_vectors[rainbow_min_value]

    rainbow_density = pd.DataFrame(np.outer(rainbow_fs, rainbow_fs), index=hf_states, columns=hf_states)
    rainbow_eam = qEAM.EAM(rainbow_density)

    # plot.eam_plot(rainbow_eam)

    plot.compare_plot(con_matrix, rainbow_eam, "8 qubits in rainbow geometry")

# #########################################################################
# Pairs of neighbours (Half-filling)
if ctl.geom_hf_8q_pairs == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    pairs = (qO.connect_op(8, [1], [2]) +
             qO.connect_op(8, [3], [4]) +
             qO.connect_op(8, [5], [6]) +
             qO.connect_op(8, [7], [8]))

    pairs_matrix = -1 * pairs.loc[hf_states, hf_states]
    con_matrix = pairs.loc[one_states, one_states]

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    pairs_values = qF.eigenvals(pairs_matrix)
    pairs_min_value = np.argmin(pairs_values)
    pairs_vectors = qF.eigenstates(pairs_matrix)
    pairs_fs = pairs_vectors[pairs_min_value]

    pairs_density = pd.DataFrame(np.outer(pairs_fs, pairs_fs), index=hf_states, columns=hf_states)
    pairs_eam = qEAM.EAM(pairs_density)

    # plot.eam_plot(pairs_eam)

    plot.compare_plot(con_matrix, pairs_eam, "8 qubits in pairs")

# #########################################################################
# 1 and 8 connect with everyone (Half-filling)
if ctl.geom_hf_8q_far == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    connect_1_8 = (qO.connect_op(8, [1], [2, 3, 4, 5, 6, 7]) +
                   qO.connect_op(8, [8], [2, 3, 4, 5, 6, 7]))

    connect_1_8_matrix = -1 * connect_1_8.loc[hf_states, hf_states]
    con_matrix = connect_1_8.loc[one_states, one_states]

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'circular')

    connect_1_8_values = qF.eigenvals(connect_1_8_matrix)
    connect_1_8_min_value = np.argmin(connect_1_8_values)
    connect_1_8_vectors = qF.eigenstates(connect_1_8_matrix)
    connect_1_8_fs = connect_1_8_vectors[connect_1_8_min_value]

    connect_1_8_density = pd.DataFrame(np.outer(connect_1_8_fs, connect_1_8_fs), index=hf_states, columns=hf_states)
    connect_1_8_eam = qEAM.EAM(connect_1_8_density)

    # plot.eam_plot(connect_1_8_eam)

    plot.compare_plot(con_matrix, connect_1_8_eam, "8 qubits - Connecting qubits 1 and 8 with the rest")

# #########################################################################
# Star inside octagon (Half-filling)
if ctl.geom_hf_8q_star == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    star_octagon = (qO.connect_op(8, [1], [2, 4, 6, 8]) +
                    qO.connect_op(8, [2], [3, 5, 7]) +
                    qO.connect_op(8, [3], [4, 6, 8]) +
                    qO.connect_op(8, [4], [5, 7]) +
                    qO.connect_op(8, [5], [6, 8]) +
                    qO.connect_op(8, [6], [7]) +
                    qO.connect_op(8, [7], [8]))

    star_matrix = -1 * star_octagon.loc[hf_states, hf_states]
    con_matrix = star_octagon.loc[one_states, one_states]

    #plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'circular')

    star_values = qF.eigenvals(star_matrix)
    star_min_value = np.argmin(star_values)
    star_vectors = qF.eigenstates(star_matrix)
    star_fs = star_vectors[star_min_value]

    star_density = pd.DataFrame(np.outer(star_fs, star_fs), index=hf_states, columns=hf_states)
    star_eam = qEAM.EAM(star_density)

    #plot.eam_plot(star_eam)

    plot.compare_plot(con_matrix, star_eam, "8 qubits in star geometry")

# #########################################################################
# Mix rainbow and neighbours (Half-filling)
if ctl.geom_hf_8q_mix == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    matrix_mix = (qO.connect_op(8, [1], [2, 8]) +
                  qO.connect_op(8, [2], [3, 7]) +
                  qO.connect_op(8, [3], [4, 6]) +
                  qO.connect_op(8, [4], [5]) +
                  qO.connect_op(8, [5], [6]) +
                  qO.connect_op(8, [6], [7]) +
                  qO.connect_op(8, [7], [8]))

    mix_matrix = -1 * matrix_mix.loc[hf_states, hf_states]
    con_matrix = matrix_mix.loc[one_states, one_states]

    #plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    mix_values = qF.eigenvals(mix_matrix)
    mix_min_value = np.argmin(mix_values)
    mix_vectors = qF.eigenstates(mix_matrix)
    mix_fs = mix_vectors[mix_min_value]

    mix_density = pd.DataFrame(np.outer(mix_fs, mix_fs), index=hf_states, columns=hf_states)
    mix_eam = qEAM.EAM(mix_density)

    #plot.eam_plot(mix_eam)

    plot.compare_plot(con_matrix, mix_eam, "8 qubits in mixed geometry")

# #########################################################################
# Let's do now this: consider two positions occupied of 8 and do the analysis, then 3 positions,
# then 4 positions, and we will compare their EAMs
if ctl.geom_chain == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    two_states = qS.generateNQubitsStates(8, 2)
    three_states = qS.generateNQubitsStates(8, 3)
    hf_states = qS.generateNQubitsStates(8, 4)

    chain_matrix = -qO.kinetic_operator(8, True)
    con_matrix = -chain_matrix.loc[one_states, one_states]

    plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    two_matrix = chain_matrix.loc[two_states, two_states]
    three_matrix = chain_matrix.loc[three_states, three_states]
    hf_matrix = chain_matrix.loc[hf_states, hf_states]

    two_values = qF.eigenvals(two_matrix)
    two_min_value = np.argmin(two_values)
    two_vectors = qF.eigenstates(two_matrix)
    two_fs = two_vectors[two_min_value]
    two_density = pd.DataFrame(np.outer(two_fs, two_fs), index=two_states, columns=two_states)
    two_eam = qEAM.EAM(two_density)

    plot.eam_plot(two_eam, True)

    three_values = qF.eigenvals(three_matrix)
    three_min_value = np.argmin(three_values)
    three_vectors = qF.eigenstates(three_matrix)
    three_fs = three_vectors[three_min_value]
    three_density = pd.DataFrame(np.outer(three_fs, three_fs), index=three_states, columns=three_states)
    three_eam = qEAM.EAM(three_density)

    plot.eam_plot(three_eam, True)

    hf_values = qF.eigenvals(hf_matrix)
    hf_min_value = np.argmin(hf_values)
    hf_vectors = qF.eigenstates(hf_matrix)
    hf_fs = hf_vectors[hf_min_value]
    hf_density = pd.DataFrame(np.outer(hf_fs, hf_fs), index=hf_states, columns=hf_states)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam, True)


# #########################################################################
# Neural (Half-filling)
if ctl.geom_hf_neural == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    neural = (qO.connect_op(8, [1], [2, 3, 4]) +
              qO.connect_op(8, [8], [5, 6, 7]) +
              qO.connect_op(8, [2], [5]) +
              qO.connect_op(8, [3], [6]) +
              qO.connect_op(8, [4], [7]))

    neural_matrix = -1 * neural.loc[hf_states, hf_states]
    con_matrix = neural.loc[one_states, one_states]

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    neural_values = qF.eigenvals(neural_matrix)
    neural_min_value = np.argmin(neural_values)
    neural_vectors = qF.eigenstates(neural_matrix)
    neural_fs = neural_vectors[neural_min_value]

    neural_density = pd.DataFrame(np.outer(neural_fs, neural_fs), index=hf_states, columns=hf_states)
    neural_eam = qEAM.EAM(neural_density)

    # plot.eam_plot(neural_eam)

    plot.compare_plot(con_matrix, neural_eam, "8 qubits - Neural")

# #########################################################################
# Neural 2 (Half-filling)
if ctl.geom_hf_neural2 == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    neural2 = (qO.connect_op(8, [1], [2, 3, 4]) +
               qO.connect_op(8, [8], [5, 6, 7]) +
               qO.connect_op(8, [2], [5, 3]) +
               qO.connect_op(8, [3], [6, 4]) +
               qO.connect_op(8, [4], [7]) +
               qO.connect_op(8, [6], [5, 7]))

    neural_matrix = -1 * neural2.loc[hf_states, hf_states]
    con_matrix = neural2.loc[one_states, one_states]

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'linear')

    neural_values = qF.eigenvals(neural_matrix)
    neural_min_value = np.argmin(neural_values)
    neural_vectors = qF.eigenstates(neural_matrix)
    neural_fs = neural_vectors[neural_min_value]

    neural_density = pd.DataFrame(np.outer(neural_fs, neural_fs), index=hf_states, columns=hf_states)
    neural_eam = qEAM.EAM(neural_density)

    # plot.eam_plot(neural_eam)

    plot.compare_plot(con_matrix, neural_eam, "8 qubits - Neural 2")

# #########################################################################
# This is [1,2,3]-[4]-[5]-[6,7,8] (Half-filling)
if ctl.geom_hf_newbridge == 1:
    one_states = qS.generateNQubitsStates(8, 1)[::-1]
    hf_states = qS.generateNQubitsStates(8, 4)

    bridge_matrix = (qO.connect_op(8, [4], [1, 2, 3]) +
                     qO.connect_op(8, [5], [6, 7, 8]) +
                     qO.connect_op(8, [4], [5]) +
                     qO.connect_op(8, [2], [3,1]) + qO.connect_op(8, [6],[8]))
    con_matrix = bridge_matrix.loc[one_states, one_states]
    bridge_matrix = -bridge_matrix.loc[hf_states, hf_states]

    # plot.eam_plot(con_matrix)
    plot.adjacency_matrix_graph(con_matrix, 'circular')

    bridge_values = qF.eigenvals(bridge_matrix)
    bridge_min_value = np.argmin(bridge_values)
    bridge_vectors = qF.eigenstates(bridge_matrix)
    bridge_fs = bridge_vectors[bridge_min_value]

    bridge_density = pd.DataFrame(np.outer(bridge_fs, bridge_fs), index=hf_states, columns=hf_states)
    bridge_eam = qEAM.EAM(bridge_density)

    # plot.eam_plot(bridge_eam)

    plot.compare_plot(con_matrix, bridge_eam, "8 qubits in new bridge geometry")

# #########################################################################
# Bridge with 12 qubits
if ctl.geom_hf_12_bridge == 1:
    mmm = qO.sparseConnectOp(12, [1], [2, 3]) +\
          qO.sparseConnectOp(12, [5], [3, 4, 6]) +\
          qO.sparseConnectOp(12, [4], [2, 6]) + \
          qO.sparseConnectOp(12, [7], [6, 8, 9]) + \
          qO.sparseConnectOp(12, [9], [8, 11]) + \
          qO.sparseConnectOp(12, [10], [8, 12]) + \
          qO.sparseConnectOp(12, [12], [11])

    ones_states = qS.nStates(12, 1)
    ones_mmm = mmm[ones_states, :].tocsc()[:, ones_states].tocsr()
    hf_states = qS.nStates(12, 6)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mmm = -mmm[hf_states, :].tocsc()[:, hf_states].tocsr()
    ones_vals, ones_vecs = qF.sparseEigen(ones_mmm)
    hf_vals, hf_vecs = qF.sparseEigen(hf_mmm)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    ones_names = qS.generateNQubitsStates(12, 1)
    ones_pd = pd.DataFrame(ones_mmm.toarray(), index=ones_names, columns=ones_names)

    #plot.eam_plot(ones_pd)

    plot.compare_plot(ones_pd, hf_eam, "12 qubits bridge")

# #########################################################################
# This is the test for the bridge one and the probs of the eigenvectors
if ctl.new_vecs == 1:
    bridge_matrix = qO.sparseConnectOp(8, [4], [1, 2, 3]) + \
                    qO.sparseConnectOp(8, [5], [6, 7, 8]) + \
                    qO.sparseConnectOp(8, [4], [5])

    ones_states = qS.nStates(8, 1)
    ones_mat = bridge_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -bridge_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    ones_vals, ones_vecs = qF.sparseEigen(ones_mat, k=4)
    probs = [vector**2 for vector in ones_vecs.T]

    plot.plotProbsNode(probs)

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam)

# #########################################################################
# This is the two points geometry with the eigenvector probs
if ctl.new_two == 1:
    two_matrix = qO.sparseConnectOp(8, [1], [2, 3, 4, 5, 6, 7]) + \
                 qO.sparseConnectOp(8, [8], [2, 3, 4, 5, 6, 7])

    ones_states = qS.nStates(8, 1)
    ones_mat = two_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -two_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    ones_vals, ones_vecs = qF.sparseEigen(ones_mat, k=4)
    probs = [vector ** 2 for vector in ones_vecs.T]

    plot.plotProbsNode(probs)

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam)

# #########################################################################
# This is the tree geometry
if ctl.geom_tree_13 == 1:
    tree_matrix = qO.sparseConnectOp(13, [1], [2, 3, 4]) + \
                  qO.sparseConnectOp(13, [2], [5, 6, 7]) + \
                  4*qO.sparseConnectOp(13, [3], [8, 9, 10]) + \
                  qO.sparseConnectOp(13, [4], [11, 12, 13])

    ones_states = qS.nStates(13, 1)[::-1]
    ones_mat = tree_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(13, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(13, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(13, 6)
    hf_mat = -tree_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)

if ctl.benzene == 1:
    benzene_matrix = 2*qO.sparseConnectOp(12, [1], [2]) + \
                     qO.sparseConnectOp(12, [2], [3]) + \
                     2*qO.sparseConnectOp(12, [3], [4]) + \
                     qO.sparseConnectOp(12, [4], [5]) + \
                     2*qO.sparseConnectOp(12, [5], [6]) + \
                     qO.sparseConnectOp(12, [6], [1]) + \
                     qO.sparseConnectOp(12, [1], [7]) + \
                     qO.sparseConnectOp(12, [2], [8]) + \
                     qO.sparseConnectOp(12, [3], [9]) + \
                     qO.sparseConnectOp(12, [4], [10]) + \
                     qO.sparseConnectOp(12, [5], [11]) + \
                     qO.sparseConnectOp(12, [6], [12])

    ones_states = qS.nStates(12, 1)[::-1]
    ones_mat = benzene_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(12, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(12, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mat = -benzene_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, False, True)


if ctl.benzene2 == 1:
    benzene_matrix = 3*qO.sparseConnectOp(12, [1], [2, 6]) + \
                     3*qO.sparseConnectOp(12, [3], [2, 4]) + \
                     3*qO.sparseConnectOp(12, [5], [4, 6]) + \
                     qO.sparseConnectOp(12, [1], [7]) + \
                     qO.sparseConnectOp(12, [2], [8]) + \
                     qO.sparseConnectOp(12, [3], [9]) + \
                     qO.sparseConnectOp(12, [4], [10]) + \
                     qO.sparseConnectOp(12, [5], [11]) + \
                     qO.sparseConnectOp(12, [6], [12])

    ones_states = qS.nStates(12, 1)[::-1]
    ones_mat = benzene_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(12, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(12, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mat = -benzene_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)

if ctl.double_ring == 1:
    double_ring_matrix = qO.sparseConnectOp(10, [1], [2]) + \
                         qO.sparseConnectOp(10, [2], [3]) + \
                         qO.sparseConnectOp(10, [3], [4]) + \
                         qO.sparseConnectOp(10, [4], [5]) + \
                         qO.sparseConnectOp(10, [5], [1]) + \
                         qO.sparseConnectOp(10, [6], [1, 7]) + \
                         qO.sparseConnectOp(10, [7], [2, 8]) + \
                         qO.sparseConnectOp(10, [8], [3, 9]) + \
                         qO.sparseConnectOp(10, [9], [4, 10]) + \
                         qO.sparseConnectOp(10, [10], [5, 6])

    ones_states = qS.nStates(10, 1)[::-1]
    ones_mat = double_ring_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(10, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(10, 5)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(10, 5)
    hf_mat = -double_ring_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)

if ctl.double_ring2 == 1:
    double_ring_matrix = 2*qO.sparseConnectOp(10, [1], [2]) + \
                         2*qO.sparseConnectOp(10, [2], [3]) + \
                         2*qO.sparseConnectOp(10, [3], [4]) + \
                         2*qO.sparseConnectOp(10, [4], [5]) + \
                         2*qO.sparseConnectOp(10, [5], [1]) + \
                         qO.sparseConnectOp(10, [6], [1, 7]) + \
                         qO.sparseConnectOp(10, [7], [2, 8]) + \
                         qO.sparseConnectOp(10, [8], [3, 9]) + \
                         qO.sparseConnectOp(10, [9], [4, 10]) + \
                         qO.sparseConnectOp(10, [10], [5, 6])

    ones_states = qS.nStates(10, 1)[::-1]
    ones_mat = double_ring_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(10, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(10, 5)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(10, 5)
    hf_mat = -double_ring_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, False, True)

if ctl.moebius == 1:
    double_ring_matrix = qO.sparseConnectOp(12, [1], [2]) + \
                         qO.sparseConnectOp(12, [2], [3, 8]) + \
                         qO.sparseConnectOp(12, [3], [4, 9]) + \
                         qO.sparseConnectOp(12, [4], [5, 10]) + \
                         qO.sparseConnectOp(12, [5], [6, 11]) + \
                         qO.sparseConnectOp(12, [6], [7]) + \
                         qO.sparseConnectOp(12, [7], [8]) + \
                         qO.sparseConnectOp(12, [8], [9]) + \
                         qO.sparseConnectOp(12, [9], [10]) + \
                         qO.sparseConnectOp(12, [10], [11]) + \
                         qO.sparseConnectOp(12, [11], [12]) + \
                         qO.sparseConnectOp(12, [12], [1])

    ones_states = qS.nStates(12, 1)[::-1]
    ones_mat = double_ring_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(12, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(12, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mat = -double_ring_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, False, True)

if ctl.acid_tio == 1:
    acid_tio_matrix = qO.sparseConnectOp(8, [4], [1, 2, 3, 5]) + \
                      qO.sparseConnectOp(8, [5, 6], [5, 6]) + \
                      qO.sparseConnectOp(8, [7], [5, 8])


    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = acid_tio_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(8, 4)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -acid_tio_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, False, True)

if ctl.ammonia == 1:
    ammonia_matrix = qO.sparseConnectOp(4, [4], [1, 2, 3])

    ones_states = qS.nStates(4, 1)[::-1]
    ones_mat = ammonia_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(4, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(4, 2)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(4, 2)
    hf_mat = -ammonia_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, False, True)

if ctl.star2 == 1:
    ammonia_matrix = qO.sparseConnectOp(8, [1], [4, 6]) + \
                     qO.sparseConnectOp(8, [2], [5, 7]) + \
                     qO.sparseConnectOp(8, [3], [6, 8]) + \
                     qO.sparseConnectOp(8, [5], [8]) + \
                     qO.sparseConnectOp(8, [4], [7])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = ammonia_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(8, 4)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -ammonia_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True, False)

# #########################################################################
# This is the tree geometry version 2
if ctl.geom_tree_13_v2 == 1:
    tree_matrix = qO.sparseConnectOp(13, [1], [2, 3, 4]) + \
                  qO.sparseConnectOp(13, [2], [5, 6, 7]) + \
                  qO.sparseConnectOp(13, [3], [8, 9, 10]) + \
                  qO.sparseConnectOp(13, [4], [11, 12, 13]) + \
                  qO.sparseConnectOp(13, [7], [8]) + \
                  qO.sparseConnectOp(13, [10], [11]) + \
                  qO.sparseConnectOp(13, [13], [5])

    ones_states = qS.nStates(13, 1)[::-1]
    ones_mat = tree_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(13, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(13, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(13, 6)
    hf_mat = -tree_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)

# #########################################################################
# This is a geometry inspired on the last
if ctl.geom_tree_inspo == 1:
    tree_inspo_matrix = qO.sparseConnectOp(10, [1], [2, 3, 4]) + \
                        qO.sparseConnectOp(10, [5], [2, 3]) + \
                        qO.sparseConnectOp(10, [6], [3, 4]) + \
                        qO.sparseConnectOp(10, [7], [2, 4]) + \
                        qO.sparseConnectOp(10, [8], [2, 5, 3]) + \
                        qO.sparseConnectOp(10, [9], [3, 6, 4]) + \
                        qO.sparseConnectOp(10, [10], [2, 4, 7])

    ones_states = qS.nStates(10, 1)[::-1]
    ones_mat = tree_inspo_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(10, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(10, 5)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(10, 5)
    hf_mat = -tree_inspo_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)


# #########################################################################
# This is pyramid tree
if ctl.pyramid_tree == 1:
    pyramid_tree_matrix = qO.sparseConnectOp(13, [1], [2, 3, 4]) + \
                          qO.sparseConnectOp(13, [2], [5, 6, 7]) + \
                          qO.sparseConnectOp(13, [3], [8, 9, 10]) + \
                          qO.sparseConnectOp(13, [4], [11, 12, 13]) + \
                          qO.sparseConnectOp(13, [2], [3, 4]) + \
                          qO.sparseConnectOp(13, [3], [4]) + \
                          qO.sparseConnectOp(13, [5], [6, 7]) + \
                          qO.sparseConnectOp(13, [6], [7]) + \
                          qO.sparseConnectOp(13, [8], [9, 10]) + \
                          qO.sparseConnectOp(13, [9], [10]) + \
                          qO.sparseConnectOp(13, [11], [12, 13]) + \
                          qO.sparseConnectOp(13, [12], [13])

    ones_states = qS.nStates(13, 1)[::-1]
    ones_mat = pyramid_tree_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(13, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(13, 6)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(13, 6)
    hf_mat = -pyramid_tree_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True)

# #########################################################################
# Let's see how does deleting connections affect
# Two points geometry and delete the connections with qubit 1
if ctl.del_two_points == 1:
    del_two_points_matrix = qO.sparseConnectOp(8, [8], [2, 3, 4, 5, 6, 7])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = del_two_points_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)
    # #print(con_matrix.to_string())
    #
    # #plot.adjacency_matrix_graph(con_matrix, 'circular')
    #
    hf_states = qS.nStates(8, 4)
    # #print(hf_states)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -del_two_points_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    #
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    plot.eam_plot(hf_eam, True, True)

# #########################################################################
# Let's see how does deleting connections affect
# Two points geometry and delete the connections with qubit 1