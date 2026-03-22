#jaccard coefficient
import networkx as nx
import pickle
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc

def recall_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return np.sum(y_true[idx]) / np.sum(y_true)

with open("all_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

for name, G in graphs.items():

    print("Dataset-", name)
    print("--------------------")

    edges = list(G.edges())
    random.shuffle(edges)

    # 90% train / 10% test split
    split = int(0.9 * len(edges))
    train_edges = edges[:split]
    test_edges = edges[split:]
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    scores = []
    y_true = []

    preds = nx.jaccard_coefficient(G_train)

    for u, v, score in preds:

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

    print("AUC:", round(auc_score,4))
    print("AUPR:", round(aupr,4))
    print("Average Precision:", round(avg_precision,4))
    print("Recall@{}:".format(k), round(recall_k,4))