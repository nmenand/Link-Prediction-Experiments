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
import copy

from ogb.linkproppred import LinkPropPredDataset



parser = argparse.ArgumentParser(description='Create Splits')
parser.add_argument("--sbm", "-s", help="Create an SBM", action="store_true")
parser.add_argument("--ogb", "-o", help="Load Dataset from OGB", action="store_true")
parser.add_argument("--ppi", "-p", help="Load Dataset from PPI", action="store_true")
parser.add_argument("source_graph", help="Input Grpah")
parser.add_argument("dataset", help="Dataset Name")
args = parser.parse_args()

dataset = args.dataset



graph_name = args.source_graph
output_name = args.dataset
block_size = 50
block_number = 200
test_size = 0.2
top_k = 0.2
density = 0.3


if args.sbm:
    g = nx.generators.community.stochastic_block_model([block_size for _ in range(block_number)], [[density if i == j else (density/(block_number*block_size)) for j in range(block_number)] for i in range(block_number)])
    nx.write_edgelist(g, f"{graph_name}.edgelist")
    g = nx.read_edgelist(f"{graph_name}.edgelist", nodetype=int)
    
    adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
    adj = sp.tril(adj, k=-1)
    print("Generated SBM")
    shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
    train_ind, test_ind = list( shuffle.split(range(adj.nnz)))[0]
    train_pair = np.vstack(adj.nonzero()).T[train_ind]
    test_pair = np.vstack(adj.nonzero()).T[test_ind] 

elif args.ogb:
    dataset = LinkPropPredDataset(name = graph_name)
    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    graph = dataset[0] # graph: library-agnostic graph object
    all_edges = np.append(train_edge['edge'], test_edge['edge'], axis=0)
    all_edges = np.append(all_edges, valid_edge['edge'], axis=0)
    g = nx.from_edgelist(all_edges)
    # Writing and reading fixes a weird networkx serialization bug 
    nx.write_edgelist(g, f"{graph_name}.edgelist")
    g = nx.read_edgelist(f"{graph_name}.edgelist", nodetype=int)

    adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
    adj = sp.tril(adj, k=-1)
    train_pair, test_pair = np.append(train_edge['edge'], valid_edge['edge'], axis=0), test_edge['edge']
    train_ind = np.array([0 for i in range(len(train_pair))])
    test_ind = np.array([0 for i in range(len(test_pair))])

    print("Converted OGB graph")

elif args.ppi:
    data = open(graph_name,'r')
    g = nx.Graph()
    e_list = []
    for line in data:
        edges = line.split()
        for i in range(0,len(edges)//3):
            (u,v) = int(edges[3*i]),int(edges[(3*i)+1])
            e_list.append((u,v))
    g.add_edges_from(e_list)

    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
    adj = sp.tril(adj, k=-1)
    shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
    train_ind, test_ind = list( shuffle.split(range(adj.nnz)))[0]
    train_pair = np.vstack(adj.nonzero()).T[train_ind]
    test_pair = np.vstack(adj.nonzero()).T[test_ind]

    numeric_indices = [index for index in range(g.number_of_nodes())]
    node_indices = sorted([node for node in g.nodes()])
    print(numeric_indices == node_indices)

else:
    g = nx.read_edgelist(graph_name, nodetype=int)
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
    adj = sp.tril(adj, k=-1)
    shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
    train_ind, test_ind = list( shuffle.split(range(adj.nnz)))[0]
    train_pair = np.vstack(adj.nonzero()).T[train_ind]
    test_pair = np.vstack(adj.nonzero()).T[test_ind]
print(f'Cluster: {nx.average_clustering(g)}')
print(f'V: {len(g.nodes())}')
print(f'E: {g.number_of_edges()}')
print(f'Tri: {sum(nx.triangles(g).values())}')
print(f'Tran: {nx.transitivity(g)}')
print(f'Mean Deg: {np.mean([g.degree(node) for node in g.nodes()])}')

train_g = copy.deepcopy(nx.Graph(g))
train_g.remove_edges_from(test_pair)
train_g.add_edges_from(train_pair)
# Writing and reading fixes a weird networkx serialization bug 
nx.write_edgelist(train_g, f"{graph_name}.edgelist")
#train_g = nx.read_edgelist(f"{graph_name}.edgelist", nodetype=int)


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

with open(f"{output_name}-G.json", "w") as out:
    out.write(json.dumps(train_g, default=nx.node_link_data))
id_map = dict()
class_map = dict()
for node in train_g.nodes():
    id_map[str(node)] = node
    class_map[str(node)] = [1, 0] 
with open(f"{output_name}-id_map.json", "w") as out:
    out.write(json.dumps(id_map))
with open(f"{output_name}-class_map.json", "w") as out:
    out.write(json.dumps(class_map))

with open(f"{output_name}-train-edges.txt", "w") as out:
    for u,v in train_pair:
        out.write(str(u)+' '+str(v)+' 1\n')
