#sbm
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

def compute_block_stats(G, partition):
    block_nodes = {}
    for node,b in partition.items():
        block_nodes.setdefault(b,set()).add(node)

    blocks = list(block_nodes.keys())
    l = {}
    r = {}

    for a in blocks:
        for b in blocks:
            nodes_a = block_nodes[a]
            nodes_b = block_nodes[b]

            if a==b:
                possible = len(nodes_a)*(len(nodes_a)-1)/2
                edges = 0
                for u in nodes_a:
                    for v in nodes_a:
                        if u<v and G.has_edge(u,v):
                            edges+=1
            else:
                possible = len(nodes_a)*len(nodes_b)
                edges = 0
                for u in nodes_a:
                    for v in nodes_b:
                        if G.has_edge(u,v):
                            edges+=1

            l[(a,b)] = edges
            r[(a,b)] = possible

    return l,r

def entropy(l,r):
    H = 0
    for key in l:
        e = l[key]
        p = r[key]
        if p==0:
            continue
        prob = e/(p+1e-9)
        if prob>0 and prob<1:
            H += p*(prob*math.log(prob)+(1-prob)*math.log(1-prob))
    return -H

with open("all_graphs.pkl","rb") as f:
    graphs = pickle.load(f)

for name,G in graphs.items():

    print("Dataset-",name)
    print("Nodes:",G.number_of_nodes())
    print("Edges:",G.number_of_edges())
    print("--------------------")

    edges=list(G.edges())
    random.shuffle(edges)

    split=int(0.9*len(edges))
    train_edges=edges[:split]
    test_edges=set(edges[split:])

    G_train=nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    nodes=list(G_train.nodes())
    n=len(nodes)

    B=int(math.sqrt(n))
    partition={node:random.randint(0,B-1) for node in nodes}

    scores_dict={}

    samples=50

    for _ in range(samples):

        node=random.choice(nodes)
        old_block=partition[node]
        new_block=random.randint(0,B-1)
        partition[node]=new_block

        l,r=compute_block_stats(G_train,partition)
        H_new=entropy(l,r)

        partition[node]=old_block
        l,r=compute_block_stats(G_train,partition)
        H_old=entropy(l,r)

        if random.random() < math.exp(-(H_new-H_old)):
            partition[node]=new_block

        l,r=compute_block_stats(G_train,partition)

        for u,v in nx.non_edges(G_train):

            bu=partition[u]
            bv=partition[v]

            edges_ab=l.get((bu,bv),0)
            poss_ab=r.get((bu,bv),1)

            score=(edges_ab+1)/(poss_ab+2)

            scores_dict.setdefault((u,v),[]).append(score)

    scores=[]
    y_true=[]

    for u,v in nx.non_edges(G_train):

        score=np.mean(scores_dict.get((u,v),[0]))
        scores.append(score)

        if (u,v) in test_edges or (v,u) in test_edges:
            y_true.append(1)
        else:
            y_true.append(0)

    scores=np.array(scores)
    y_true=np.array(y_true)

    auc_score=roc_auc_score(y_true,scores)

    avg_precision=average_precision_score(y_true,scores)

    precision,recall,_=precision_recall_curve(y_true,scores)
    aupr=auc(recall,precision)

    k=len(test_edges)
    recall_k=recall_at_k(y_true,scores,k)

    print("AUC:",round(auc_score,4))
    print("AUPR:",round(aupr,4))
    print("Average Precision:",round(avg_precision,4))
    print("Recall@{}:".format(k),round(recall_k,4))