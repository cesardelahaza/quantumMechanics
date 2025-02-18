import matplotlib.pyplot as plt
import pandas as pd


def eam_plot(mat: pd.DataFrame, addColorbar:bool = False):
    n_qubits = len(mat.columns)
    fig, ax = plt.subplots()
    pl = ax.matshow(mat, cmap='cividis')
    plt.xticks(range(n_qubits))
    plt.yticks(range(n_qubits))
    ax.set_xticklabels([str(i) for i in range(1, n_qubits+1)])
    ax.set_yticklabels([str(i) for i in range(1, n_qubits+1)])
    if addColorbar:
        fig.colorbar(pl)
    plt.show()


def compare_plot(con_mat, eam_mat, title:str):
    n_qubits = len(con_mat.columns)
    fig, (fig1, fig2) = plt.subplots(1, 2)
    fig.suptitle(title)
    p1 = fig1.matshow(con_mat, cmap='cividis')
    fig1.set_title("Connectivity")
    fig1.set_xticks(range(n_qubits))
    fig1.set_yticks(range(n_qubits))
    fig1.set_xticklabels([str(i) for i in range(1, n_qubits + 1)])
    fig1.set_yticklabels([str(i) for i in range(1, n_qubits + 1)])
    fig.colorbar(p1, pad=0.02, location='bottom')
    p2 = fig2.matshow(eam_mat, cmap='cividis')
    fig2.set_title("EAM")
    fig2.set_xticks(range(n_qubits))
    fig2.set_yticks(range(n_qubits))
    fig2.set_xticklabels([str(i) for i in range(1, n_qubits + 1)])
    fig2.set_yticklabels([str(i) for i in range(1, n_qubits + 1)])
    fig.colorbar(p2, pad=0.02, location='bottom')
    plt.show()
