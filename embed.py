import getopt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import random
import pickle as pkl

from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from karateclub import BoostNE, NodeSketch, GraRep, NetMF, Node2Vec, DeepWalk, RandNE, GLEE, Walklets, Role2Vec, GraphWave

import argparse

parser = argparse.ArgumentParser(description='Create Graph Embeddings')
parser.add_argument("--graRep", "-g", help="Use GraRep; only use with small datasets", action="store_true")
parser.add_argument("dataset", help="Dataset Name")
parser.add_argument("dim", help="number of dimensions", type=int)
args = parser.parse_args()

dataset = args.dataset
dim = args.dim


if args.graRep:
  embedding_names = ['node2vec', 'deepwalk', 'netmf', 'GraRep', 'BoostNE', 'RandNE', 'Walklets', 'Role2Vec']
else:
  embedding_names = ['node2vec', 'deepwalk', 'netmf', 'BoostNE', 'RandNE', 'Walklets', 'Role2Vec']


train_g = nx.read_adjlist(f"{dataset}-train.adjlist", nodetype=int)
#train_g = nx.convert_node_labels_to_integers(train_g)

for emb in embedding_names:
    print(emb)
    if emb == 'node2vec':
        model = Node2Vec(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
        #vecs = Node2Vec(adj)
    elif emb == 'deepwalk':
        model = DeepWalk(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
        #vecs = DeepWalk(adj)
    elif emb == 'netmf':
        model = NetMF(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
        #vecs = NetMF(adj)
    elif emb == 'BoostNE':
        model = BoostNE(dimensions=dim, iterations=8)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'GraRep':
        model = GraRep(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'NodeSketch':
        model = NodeSketch(dimensions=dim, iterations=6, decay=0.01)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'GLEE':
        model = GLEE(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'RandNE':
        model = RandNE(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'Walklets':
        model = Walklets(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
    elif emb == 'Role2Vec':
        model = Role2Vec(dimensions=dim)
        model.fit(train_g)
        vecs = model.get_embedding()
    np.save(f'{emb}-{dataset}.npy', vecs)
