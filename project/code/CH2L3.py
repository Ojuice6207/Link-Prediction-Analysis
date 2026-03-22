#ch2l3
import networkx as nx
import pickle
import numpy as np
import random
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc

def recall_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return np.sum(y_true[idx]) / np.sum(y_true)

with open("all_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

for name,G in graphs.items():
    print("Dataset-",name)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("--------------------")

    edges = list(G.edges())
    random.shuffle(edges)
    split = int(0.9*len(edges))
    train_edges = edges[:split]
    test_edges = set(edges[split:])

    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    scores = []
    y_true = []

    for x,y in nx.non_edges(G_train):
        Nx = set(G_train.neighbors(x))
        Ny = set(G_train.neighbors(y))
        tri_length = set()
        for u in Nx:
            for v in Ny:
                if G_train.has_edge(u,v):
                    tri_length.add(u)
                    tri_length.add(v)
        score = 0
        for z1 in Nx:
            for z2 in Ny:
                if G_train.has_edge(z1,z2):

                    nbhr1 = set(G_train.neighbors(z1))
                    nbhr2 = set(G_train.neighbors(z2))

                    c1 = o1 = c2 = o2 = 0

                    for u in nbhr1:
                        if u in tri_length:
                            c1 += 1
                        elif u!=x and u!=y:
                            o1 += 1

                    for u in nbhr2:
                        if u in tri_length:
                            c2 += 1
                        elif u!=x and u!=y:
                            o2 += 1

                    score += math.sqrt((1+c1)*(1+c2))/math.sqrt((1+o1)*(1+o2))

        scores.append(score)

        if (x,y) in test_edges or (y,x) in test_edges:
            y_true.append(1)
        else:
            y_true.append(0)
    scores = np.array(scores)
    y_true = np.array(y_true)

    auc_score = roc_auc_score(y_true, scores)

    avg_precision = average_precision_score(y_true, scores)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    aupr = auc(recall, precision)

    k = len(test_edges)
    recall_k = recall_at_k(y_true, scores, k)

    print("AUC:", round(auc_score,4))
    print("AUPR:", round(aupr,4))
    print("Average Precision:", round(avg_precision,4))
    print("Recall@{}:".format(k), round(recall_k,4))