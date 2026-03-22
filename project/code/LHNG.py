#LHNG

import networkx as nx
import pickle
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc

eps = 1e-6

def recall_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return np.sum(y_true[idx]) / np.sum(y_true)

with open("all_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

for name, G in graphs.items():

    print("Dataset-", name)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("--------------------")

    edges = list(G.edges())
    random.shuffle(edges)

    # 90/10 split
    split = int(0.9 * len(edges))
    train_edges = edges[:split]
    test_edges = set(edges[split:])
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    nodes = list(G_train.nodes())
    n = len(nodes)
    A = nx.to_numpy_array(G_train, nodelist=nodes)

    degrees = np.array([G_train.degree(v) for v in nodes], dtype=float)
    degrees[degrees == 0] = 1

    D = np.diag(1 / degrees)
    lambda_1 = max((np.linalg.eigvals(A)).real)

    alpha = 0.85 / lambda_1

    X = np.zeros((n, n))
    I = np.eye(n)

    # LHNG Iterations
    for _ in range(100):
        Xn = alpha * A @ X + I
        if np.linalg.norm(Xn - X) < eps:
            break
        X = Xn

    S = D @ X @ D

    scores = []
    y_true = []

    node_index = {node: i for i, node in enumerate(nodes)}

    for u, v in nx.non_edges(G_train):

        i = node_index[u]
        j = node_index[v]

        score = S[i, j]
        scores.append(score)

        if (u, v) in test_edges or (v, u) in test_edges:
            y_true.append(1)
        else:
            y_true.append(0)

    scores = np.array(scores)
    y_true = np.array(y_true)

    # AUC
    auc_score = roc_auc_score(y_true, scores)

    # Average Precision
    avg_precision = average_precision_score(y_true, scores)

    # AUPR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    aupr = auc(recall, precision)

    # Recall@K
    k = len(test_edges)
    recall_k = recall_at_k(y_true, scores, k)

    print("AUC:", round(auc_score, 4))
    print("AUPR:", round(aupr, 4))
    print("Average Precision:", round(avg_precision, 4))
    print("Recall@{}:".format(k), round(recall_k, 4))