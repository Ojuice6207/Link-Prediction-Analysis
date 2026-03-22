import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
with open("all_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

for name,G in graphs.items():
    print("Dataset-",name)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("--------------------")

    X = []
    y = []

    for u,v in nx.non_edges(G):

        cn = len(list(nx.common_neighbors(G,u,v)))
        pa = G.degree(u) * G.degree(v)

        try:
            jaccard = list(nx.jaccard_coefficient(G, [(u,v)]))[0][2]
        except:
            jaccard = 0

        X.append([cn, jaccard, pa])
        y.append(0)

    for u,v in G.edges():

        cn = len(list(nx.common_neighbors(G,u,v)))
        pa = G.degree(u) * G.degree(v)

        try:
            jaccard = list(nx.jaccard_coefficient(G, [(u,v)]))[0][2]
        except:
            jaccard = 0

        X.append([cn, jaccard, pa])
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    pred = model.predict_proba(X_test)[:,1]

    # Evaluate
    auc = roc_auc_score(y_test, pred)
    print("AUC:", auc)