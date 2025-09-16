import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.patches import Arc
import time


def eam_plot(mat: pd.DataFrame, addColorbar: bool = True, addValues: bool = False):
    n_qubits = len(mat.columns)
    fig, ax = plt.subplots()
    pl = ax.matshow(mat, cmap='BuPu')
    plt.xticks(range(n_qubits))
    plt.yticks(range(n_qubits))
    ax.set_xticklabels([str(i) for i in range(1, n_qubits+1)])
    ax.set_yticklabels([str(i) for i in range(1, n_qubits+1)])
    if addColorbar:
        fig.colorbar(pl)

    if addValues:
        for (i, j), z in np.ndenumerate(mat):
            if z > 0:
                ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize='x-small')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('/Users/cesarots/Downloads/' + timestr + '.pdf', pad_inches=0, bbox_inches='tight', format='pdf')
    # https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
    # for next time


def compare_plot(con_mat, eam_mat, title: str):
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
    fig.tight_layout()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('/Users/cesarots/Downloads/' + timestr + '.pdf', pad_inches=0, bbox_inches='tight', format='pdf')


def adjacency_matrix_graph(adjacency_matrix, layout):
    ll = len(list(adjacency_matrix.columns))
    lay = {}
    if layout == 'linear':
        lay = {i: (i, 0) for i in range(ll)}
    elif layout == 'circular':
        lay = {i: (np.cos(2*i*np.pi/ll), np.sin(2*i*np.pi/ll)) for i in range(ll)}

    G = nx.from_numpy_array(np.array(adjacency_matrix))
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, lay, ax=ax, node_color="lightblue")
    nx.draw_networkx_labels(G, lay, ax=ax, labels={i-1: str(i) for i in range(1,ll+1)})

    for edge in G.edges():
        x1, y1 = lay[edge[0]]
        x2, y2 = lay[edge[1]]

        if layout == 'linear':
            xc, yc = (x1+x2)/2, (y1+y2)/2

            angle = np.arctan2(y2-y1, x2-x1)*180/np.pi

            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            arc = Arc((xc, yc), width=dist, height=dist, angle=angle, theta1=0, theta2=180, color="black")
            ax.add_patch(arc)
        elif layout == 'circular':
            nx.draw_networkx_edges(G, lay, edgelist=[edge], ax=ax)
    plt.axis('off')
    fig.tight_layout()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('/Users/cesarots/Downloads/' + timestr + '.pdf', pad_inches=0, bbox_inches='tight', format='pdf')


def plotProbsNode(vectors, vals):
    xxx = len(vectors[0])
    x = np.arange(1, xxx + 1)

    fig, ax = plt.subplots()

    for i, vector in enumerate(vectors):
        ax.plot(x, vector, marker='o', label=f'Eig $\\lambda_{i+1}$ = {vals[i]:.3e}')

    # for xi, vi in zip(x, vectors[0]):
    #     ax.annotate(f"{xi}", (xi, vi-0.002), ha="center")

    plt.xlabel('Nodos')
    plt.ylabel('$|v_i|^2$')
    plt.title('Probabilidad de ocupación')
    plt.legend()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('/Users/cesarots/Downloads/' + timestr + '.pdf', pad_inches=0, bbox_inches='tight', format='pdf')

