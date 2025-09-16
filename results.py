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
########################################################################
# MAIN: control
########################################################################
if ctl.r_bridge_8q == 1:

    bridge_8q_matrix = qO.sparseConnectOp(8, [4], [1, 2, 3]) + \
        qO.sparseConnectOp(8, [5], [6, 7, 8]) + \
        qO.sparseConnectOp(8, [4], [5]) #+ \
        #qO.sparseChemical_potential_operator(8)

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = bridge_8q_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = -pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    #print(con_matrix)
    ones_vals, ones_vecs = qF.sparseEigen(ones_mat, k=4)
    probs = [vector ** 2 for vector in ones_vecs.T]
    #plot.plotProbsNode(probs, ones_vals)
    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -bridge_8q_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    #print(f"Eigenvals for bridge geometry: {hf_vals}")
    #print(f"Eigenvector for bridge geometry: {hf_vecs}")
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con = new_con/np.linalg.norm(new_con)
    new_hf = hf_eam/np.linalg.norm(hf_eam) #qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    #print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))

if ctl.r_twoT_8q == 1:
    twoT_8q_matrix = qO.sparseConnectOp(8, [1], [2, 3, 4, 5, 6, 7]) + \
                       qO.sparseConnectOp(8, [8], [2, 3, 4, 5, 6, 7])  # + \
    # qO.sparseChemical_potential_operator(8)

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = twoT_8q_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)

    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -twoT_8q_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    #print(f"Eigenvals for two towers geometry: {hf_vals}")
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con = new_con / np.linalg.norm(new_con)
    new_hf = hf_eam / np.linalg.norm(hf_eam)  # qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    # print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))

if ctl.r_tunnel_12q == 1:
    tunnel_mat = qO.sparseConnectOp(12, [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12])

    ones_states = qS.nStates(12, 1)[::-1]
    ones_mmm = tunnel_mat[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(12, 1)

    ones_pd = pd.DataFrame(ones_mmm.toarray(), index=ones_names, columns=ones_names)
    #plot.eam_plot(ones_pd)

    hf_states = qS.nStates(12, 6)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mmm = -tunnel_mat[hf_states, :].tocsc()[:, hf_states].tocsr()

    ones_vals, ones_vecs = qF.sparseEigen(ones_mmm)
    hf_vals, hf_vecs = qF.sparseEigen(hf_mmm)
    #print(f"Eigenvals for tunnel geometry: {hf_vals}")
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    # plot.eam_plot(hf_eam, True, True)

    new_con = ones_pd
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    new_hf = qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))

if ctl.r_star_8q == 1:
    star_8q_matrix = qO.sparseConnectOp(8, [1], [2, 4]) + \
                     qO.sparseConnectOp(8, [2], [3, 5]) + \
                     qO.sparseConnectOp(8, [3], [4, 6]) + \
                     qO.sparseConnectOp(8, [4], [5, 7]) + \
                     qO.sparseConnectOp(8, [5], [6, 8]) + \
                     qO.sparseConnectOp(8, [6], [7, 1]) + \
                     qO.sparseConnectOp(8, [7], [8, 2]) + \
                     qO.sparseConnectOp(8, [8], [1, 3])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = star_8q_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    eigvals, eigvecs = qF.sparseEigen(-ones_mat, k=3)

    #print(f"Eigenvector of star geom:\n{eigvecs}")
    #print(f"Eigenvalue of star geom: \n{eigvals}")
    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -star_8q_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    #print(f"Eigenvals for star geometry: {hf_vals}")
    #print(f"Eigenvecs for star geometry: {hf_vecs}")
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8]
    #print(new_con - hf_eam / np.max(hf_eam))

    new_hf = qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))

if ctl.r_benzene == 1:
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
    #plot.eam_plot(con_matrix, True, True)

    hf_states = qS.nStates(12, 6)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mat = -benzene_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    #print(f"Eigenvals for benzene geometry: {hf_vals}")
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    #print(new_con - hf_eam / np.max(hf_eam))

    new_hf = qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con/2 - new_hf))
    print(np.linalg.norm(new_con/2 - new_hf) / np.linalg.norm(new_con))

if ctl.r_tree_13q == 1:
    tree_matrix = qO.sparseConnectOp(13, [1], [2, 3, 4]) + \
                  qO.sparseConnectOp(13, [2], [5, 6, 7]) + \
                  qO.sparseConnectOp(13, [3], [8, 9, 10]) + \
                  qO.sparseConnectOp(13, [4], [11, 12, 13])

    ones_states = qS.nStates(13, 1)[::-1]
    ones_mat = tree_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(13, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    eigvals, eigvecs = qF.sparseEigen(-ones_mat)

    #print(f"Eigenvector of tree geom:\n{eigvecs}")
    #print(f"Eigenvalue of tree geom: \n{eigvals}")
    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(13, 6)
    hf_names = qS.generateNQubitsStates(13, 6)
    hf_mat = -tree_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    #print(f"Eigenvals for tree geometry: {hf_vals}")

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    new_hf = qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))

if ctl.r_rainbow_8q == 1:
    rainbow_mat = qO.sparseConnectOp(8, [1], [8]) + \
                  qO.sparseConnectOp(8, [2], [7]) + \
                  qO.sparseConnectOp(8, [3], [6]) + \
                  qO.sparseConnectOp(8, [4], [5]) #+ \
                  #qO.sparseChemical_potential_operator(8)

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = rainbow_mat[ones_states, :].tocsc()[:, ones_states].tocsr()

    ones_names = qS.generateNQubitsStates(8, 1)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -rainbow_mat[hf_states, :].tocsc()[:, hf_states].tocsr()

    ones_vals, ones_vecs = qF.sparseEigen(ones_mat, k=4)

    probs = [vector ** 2 for vector in ones_vecs.T]

    plot.plotProbsNode(probs, ones_vals)

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam, addValues=True)

if ctl.r_rainbow_8q_new == 1:
    rainbow_mat = 1*qO.sparseConnectOp(8, [4], [5]) + \
        0.1*qO.sparseConnectOp(8, [3], [4]) + \
        0.1*qO.sparseConnectOp(8, [5], [6]) + \
        (0.1**3)*qO.sparseConnectOp(8, [2], [3]) + \
        (0.1**3)*qO.sparseConnectOp(8, [6], [7]) + \
        (0.1**5)*qO.sparseConnectOp(8, [1], [2]) + \
        (0.1**5)*qO.sparseConnectOp(8, [7], [8])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = rainbow_mat[ones_states, :].tocsc()[:, ones_states].tocsr()
    print(-ones_mat.toarray())
    ones_names = qS.generateNQubitsStates(8, 1)
    np.savetxt("matrizRainbow8q.txt", -ones_mat.toarray(), fmt="%.6f")
    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -rainbow_mat[hf_states, :].tocsc()[:, hf_states].tocsr()
    print(hf_mat.toarray())
    ones_vals, ones_vecs = qF.sparseEigen(ones_mat, k=4)

    probs = [vector ** 2 for vector in ones_vecs.T]

    plot.plotProbsNode(probs, ones_vals)

    #hf_vals, hf_vecs = qF.sparseEigen(hf_mat, k=4)
    #print(hf_vals)
    #print(hf_vecs)
    #print(qF.eigenvals(hf_mat.toarray()))
    #print(qF.eigenstates(hf_mat.toarray()).T)
    # hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    # hf_eam = qEAM.EAM(hf_density)
    #
    # plot.eam_plot(hf_eam, addValues=True)

if ctl.r_closedRing_4 == 1:
    cR4 = qO.sparseConnectOp(6, [1], [2]) + \
          qO.sparseConnectOp(6, [2], [3]) + \
          qO.sparseConnectOp(6, [3], [4]) + \
          qO.sparseConnectOp(6, [4], [5]) + \
          qO.sparseConnectOp(6, [5], [6]) + \
          qO.sparseConnectOp(6, [6], [1]) #+ \
          #qO.sparseConnectOp(6, [7], [1])

    ones_states = qS.nStates(6, 1)[::-1]
    ones_mat = cR4[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(6, 1)

    hf_states = qS.nStates(6, 3)
    hf_names = qS.generateNQubitsStates(6, 3)
    hf_mat = -cR4[hf_states, :].tocsc()[:, hf_states].tocsr()

    #hf_vals, hf_vecs = qF.sparseEigen(hf_mat, k=6)

    ones_vals, ones_vecs = qF.eigenvals(-ones_mat.toarray()), qF.eigenstates(-ones_mat.toarray())
    print(ones_vals)
    print(ones_vecs.T)

if ctl.newcase == 1:
    case = qO.sparseConnectOp(4, [1], [2, 3]) + \
           qO.sparseConnectOp(4, [2], [3]) + \
           qO.sparseConnectOp(4, [3], [4])

    total_mat = pd.DataFrame(-case.toarray(), index=qS.generateAllPossibleStates(4), columns=qS.generateAllPossibleStates(4))
    np.savetxt('/Users/cesarots/Downloads/ejemplo.txt', total_mat, fmt='%1.3f')
    print(total_mat)
    ones_states = qS.nStates(4, 1)[::-1]
    ones_mat = case[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(4, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(4, 2)
    hf_names = qS.generateNQubitsStates(4, 2)
    hf_mat = -case[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    # print(hf_vals)
    # print(hf_vecs)
    # print(qF.eigenvals(hf_mat.toarray()))
    # print(qF.eigenstates(hf_mat.toarray()).T)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #
    #plot.eam_plot(hf_eam, addValues=True)

if ctl.chain == 1:
    mat = 2*qO.sparseConnectOp(8, [1], [2]) + \
          2*qO.sparseConnectOp(8, [2], [3]) + \
          2*qO.sparseConnectOp(8, [3], [4]) + \
          2*qO.sparseConnectOp(8, [4], [5]) + \
          2*qO.sparseConnectOp(8, [5], [6]) + \
          2*qO.sparseConnectOp(8, [6], [7]) + \
          2*qO.sparseConnectOp(8, [7], [8]) + qO.sparseConnectOp(8, [8], [1])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = mat[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -mat[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)
    #plot.eam_plot(hf_eam, addValues=True)

    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8]
    new_con = new_con / np.linalg.norm(new_con)
    new_hf = hf_eam / np.linalg.norm(hf_eam)  # qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    # print(np.linalg.norm(new_con - new_hf) / np.linalg.norm(new_con))


if ctl.plane == 1:
    mat = qO.sparseConnectOp(12, [1], [2, 5]) +\
          qO.sparseConnectOp(12, [2], [3, 6]) +\
          qO.sparseConnectOp(12, [3], [4, 7]) +\
          qO.sparseConnectOp(12, [4], [8]) +\
          qO.sparseConnectOp(12, [5], [6, 9]) +\
          qO.sparseConnectOp(12, [6], [7, 10]) +\
          qO.sparseConnectOp(12, [7], [8, 11]) +\
          qO.sparseConnectOp(12, [8], [12]) +\
          qO.sparseConnectOp(12, [10], [9, 11]) +\
          qO.sparseConnectOp(12, [11], [12])

    ones_states = qS.nStates(12, 1)[::-1]
    ones_mat = mat[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(12, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    #plot.eam_plot(con_matrix, True)

    hf_states = qS.nStates(12, 6)
    hf_names = qS.generateNQubitsStates(12, 6)
    hf_mat = -mat[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, addValues=True)
    new_con = con_matrix
    new_con.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    new_con.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    new_hf = qF.convertMatrix(hf_eam / np.max(hf_eam), 1 / 2)
    # print(new_con - hf_eam / np.max(hf_eam))
    print(np.linalg.norm(new_con - new_hf))
    print(np.linalg.norm(new_con-new_hf)/np.linalg.norm(new_con))

if ctl.r_bridge_8q_blog == 1:

    bridge_8q_matrix = qO.sparseConnectOp(8, [4], [1, 2, 3]) + \
        qO.sparseConnectOp(8, [5], [6, 7, 8]) + \
        qO.sparseConnectOp(8, [4], [5])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = bridge_8q_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = -pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    #plot.eam_plot(-con_matrix, False, False)

    hf_states = qS.nStates(8, 4)

    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -bridge_8q_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()
    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    #plot.eam_plot(hf_eam, True, False)


if ctl.chain_blog == 1:
    chain_8q = qO.sparseConnectOp(8, [1], [2, 8]) + \
    qO.sparseConnectOp(8, [3], [2, 4]) + \
    qO.sparseConnectOp(8, [5], [4, 6]) + \
    qO.sparseConnectOp(8, [7], [6, 8])

    ones_states = qS.nStates(8, 1)[::-1]
    ones_mat = chain_8q[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(8, 1)

    con_matrix = -pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)
    plot.eam_plot(-con_matrix, False, False)

    hf_states = qS.nStates(8, 4)
    hf_names = qS.generateNQubitsStates(8, 4)
    hf_mat = -chain_8q[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)

    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam, True, False)

if ctl.r_tree_13q_blog == 1:
    tree_matrix = qO.sparseConnectOp(13, [1], [2, 3, 4]) + \
                  qO.sparseConnectOp(13, [2], [5, 6, 7]) + \
                  qO.sparseConnectOp(13, [3], [8, 9, 10]) + \
                  qO.sparseConnectOp(13, [4], [11, 12, 13])

    ones_states = qS.nStates(13, 1)[::-1]
    ones_mat = tree_matrix[ones_states, :].tocsc()[:, ones_states].tocsr()
    ones_names = qS.generateNQubitsStates(13, 1)

    con_matrix = pd.DataFrame(ones_mat.toarray(), index=ones_names, columns=ones_names)

    plot.eam_plot(con_matrix, False, False)

    hf_states = qS.nStates(13, 6)
    hf_names = qS.generateNQubitsStates(13, 6)
    hf_mat = -tree_matrix[hf_states, :].tocsc()[:, hf_states].tocsr()

    hf_vals, hf_vecs = qF.sparseEigen(hf_mat)
    hf_density = pd.DataFrame(np.outer(hf_vecs, hf_vecs), index=hf_names, columns=hf_names)
    hf_eam = qEAM.EAM(hf_density)

    plot.eam_plot(hf_eam, True, False)