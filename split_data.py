import getopt
from sys import argv
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import networkx as nx
import random
from  scipy.sparse import coo_matrix
from sklearn.model_selection import ShuffleSplit
import argparse
import json



parser = argparse.ArgumentParser(description='Create Splits')
parser.add_argument("--SBM", "-s", help="Create an SBM", action="store_true")
parser.add_argument("source_graph", help="number of dimensions", type=int)
parser.add_argument("dataset", help="Dataset Name")
args = parser.parse_args()

dataset = args.dataset
dim = args.dim


graph_name = args.source_graph
output_name = args.dataset
block_size = 50
block_number = 200
test_size = 0.2
top_k = 0.2
density = 0.3


if args.sbm:
    g = nx.generators.community.stochastic_block_model([block_size for _ in range(block_number)], [[density if i == j else (density/(block_number*block_size)) for j in range(block_number)] for i in range(block_number)])
    adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
    adj = sp.tril(adj, k=-1)
    print("Generated Graph")
    shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
    train_ind, test_ind = list( shuffle.split(range(adj.nnz)))[0]
    train_pair = np.vstack(adj.nonzero()).T[train_ind]
    test_pair = np.vstack(adj.nonzero()).T[test_ind] 
else:
    g = nx.read_edgelist(graph_name, nodetype=int)
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_matrix(g)
    adj = sp.tril(adj, k=-1)
    shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
    train_ind, test_ind = list( shuffle.split(range(adj.nnz)))[0]
    train_pair = np.vstack(adj.nonzero()).T[train_ind]
    test_pair = np.vstack(adj.nonzero()).T[test_ind]
    print(np.mean([g.degree(node) for node in g.nodes()]))
    print(nx.average_clustering(g))
    print(g.number_of_edges())
train_g = g.copy()
train_g.remove_edges_from(test_pair)
print("Created Splits")
np.save(f'{output_name}-train-pair.npy', train_pair)
np.save(f'{output_name}-test-pair.npy', test_pair)
np.save(f'{output_name}-train-ind.npy', train_ind)
np.save(f'{output_name}-test-ind.npy', test_ind)
nx.write_edgelist(g, f"{output_name}.edgelist")
nx.write_edgelist(train_g, f"{output_name}-train.edgelist")
nx.write_adjlist(g, f"{output_name}.adjlist")
nx.write_adjlist(train_g, f"{output_name}-train.adjlist")

test_map = dict(zip(train_g.nodes(),[node in test_pair[0] or node in test_pair[1] for node in train_g.nodes()]))
nx.set_node_attributes(train_g, test_map, 'test')
nx.set_node_attributes(train_g, False,'val' )
json_G = nx.node_link_data(train_g)

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj

with open(f"{output_name}-G.json", "w") as out:
    out.write(json.dumps(json_G, default=serialize_sets))
id_map = dict()
class_map = dict()
for node in train_g.nodes():
    id_map[str(node)] = node
    class_map[str(node)] = [1, 0] 
with open(f"{output_name}-id_map.json", "w") as out:
    out.write(json.dumps(id_map))
with open(f"{output_name}-class_map.json", "w") as out:
    out.write(json.dumps(class_map))

