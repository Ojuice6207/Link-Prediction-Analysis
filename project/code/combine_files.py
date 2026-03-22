import networkx as nx
import pickle
import os

graphs = {}

# read all gml files
for file in os.listdir():
    if file.endswith(".gml"):
        print("Loading:", file)
        G = nx.read_gml(file,label='id')
        graphs[file] = G

# save everything into one pickle file
with open("all_graphs.pkl", "wb") as f:
    pickle.dump(graphs, f)

print("All graphs saved to all_graphs.pkl")