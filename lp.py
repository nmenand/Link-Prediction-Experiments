import getopt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import random
import copy
import argparse
import torch

from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, top_k_accuracy_score
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ogb.linkproppred import Evaluator, LinkPropPredDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import PrecisionRecallDisplay


parser = argparse.ArgumentParser(description='Evaluate embeddings')
parser.add_argument("--sbm", "-s", help="Input is from an sbm", action="store_true")
parser.add_argument("--ogb", "-o", help="Input is from OGB", action="store_true")
parser.add_argument("--ppi", "-p", help="Input is from PPI", action="store_true")
parser.add_argument("--Eval_all_edges", "-e", help="Evaluate embeddings for train + test edges", action="store_true")
parser.add_argument("--Multigraph", "-m", help="Don't filter train edges from predictions", action="store_true")
parser.add_argument("--graRep", "-g", help="Use GraRep; only use with small datasets", action="store_true")
parser.add_argument("--source_file", '-f', action='store', help="Graph Source File")
parser.add_argument("dataset", help="Dataset Name")
parser.add_argument("hits_k", help="Hits@k (k=20 or 50)", type=int)
args = parser.parse_args()

dataset = args.dataset

if args.graRep:
  embedding_names = ['cn', 'aa', 'ra', 'node2vec', 'deepwalk', 'netmf', 'GraRep',  'RandNE', 'Walklets','BoostNE', 'Role2Vec', 'graphsage-lstm', 'graphsage-mean', 'graphsage-pool', 'hpe', 'hoprec']
else:
  embedding_names = ['cn', 'aa', 'ra','node2vec', 'deepwalk', 'netmf', 'RandNE', 'Walklets', 'BoostNE', 'Role2Vec', 'graphsage-lstm', 'graphsage-mean', 'graphsage-pool', 'hpe', 'hoprec']

plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
fig_roc, ax_roc = plt.subplots()
fig_pr, ax_pr = plt.subplots()
fig_rel_10, ax_rel_10 = plt.subplots()
fig_rel_50, ax_rel_50 = plt.subplots()
fig_rel_all, ax_rel_all = plt.subplots()
fig_rel_20, ax_rel_20 = plt.subplots()
fig_hits, ax_hits = plt.subplots()
fig_rel_d, ax_rel_d = plt.subplots()
fig_sig, ax_sig = plt.subplots()

fig_ndcg_10, ax_ndcg_10 = plt.subplots()
fig_ndcg_50, ax_ndcg_50 = plt.subplots()
fig_ndcg_d, ax_ndcg_d = plt.subplots()

def sample_rows(graph, num_rows):
    """
    Provide coordinates for all entries in num_rows number of random rows

    :param graph: CsrGraph to choose rows from.
    :param num_rows: Number of rows to sample.
    """
    rows = np.random.randint(0, high=graph.number_of_nodes(), size=num_rows, dtype=np.int32)
    pairs = np.vstack([np.array([u, v]) for u in graph.nodes() for v in rows])
    return pairs

def histofy(ps, num_bins=1000):
    bins = np.zeros(num_bins, dtype=np.float32)
    for p in ps:
        bins[int( np.floor( p * (num_bins - 1)) )] += 1
    accumulator = 0
    for i in range(num_bins):
        accumulator += bins[-(i + 1)]
        bins[-(i + 1)] = accumulator / len(ps)
    #print(bins)
    return bins

def link_prediction(vecs, train_pair, test_pair, y_train, y_test, whole_adj, label_counts, emb, sample_verts, g, train_adj, test_adj, hits_test_pair, hits_neg, eval_all, multigraph):
    #X_train = np.concatenate((vecs[train_pair[:, 0]], vecs[train_pair[:, 1]]), axis=1)
    #X_test = np.concatenate((vecs[test_pair[:, 0]], vecs[test_pair[:, 1]]), axis=1)
    X_train = np.multiply(vecs[train_pair[:, 0]], vecs[train_pair[:, 1]])
    X_test = np.multiply(vecs[test_pair[:, 0]], vecs[test_pair[:, 1]])
    #X_train = np.sum(X_train, axis=1).reshape(-1,1) 
    #X_test = np.sum(X_test, axis=1).reshape(-1,1)
    
    #clf = MLPClassifier(random_state=1, hidden_layer_sizes=(256, 256, 256), learning_rate_init=0.005, max_iter=1000)
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", multi_class="ovr"))
    print('Training model')
    clf.fit(X_train, y_train.ravel())
    clf.fit(X_train, y_train)
    print('Done training')
   
    test_scores = clf.predict_proba(X_test)[:,1]
    
    #print(np.shape(X_test))
    #print(np.shape(y_test))
    print("ROC AUC SCORE: %f" % roc_auc_score(y_test, test_scores))
    p, r, t = precision_recall_curve(y_test, test_scores)
    print("PR AUC SCORE: %f" % auc(r, p))

    #X_ogb_test_true = np.concatenate((vecs[hits_test_pair[:, 0]], vecs[hits_test_pair[:, 1]]), axis=1)
    #X_ogb_test_false = np.concatenate((vecs[hits_neg[:, 0]], vecs[hits_neg[:, 1]]), axis=1)
    X_ogb_test_true  = np.multiply(vecs[hits_test_pair[:, 0]], vecs[hits_test_pair[:, 1]])
    X_ogb_test_false  = np.multiply(vecs[hits_neg[:, 0]], vecs[hits_neg[:, 1]]) 
    
    y_pos_scores = clf.predict_proba(X_ogb_test_true)[:,1]
    y_neg_scores = clf.predict_proba(X_ogb_test_false)[:,1]
    
    result = evaluator.eval({'y_pred_pos': y_pos_scores, 'y_pred_neg': y_neg_scores})
    print(f'OGB Mean Hits@{args.hits_k} Score: {result[f"hits@{args.hits_k}"]}')

    vec_precision_10 = 0
    vec_precision_20 = 0  
    vec_precision_50 = 0
    vec_precision_d = 0
    
    ndcg_10_list = []
    ndcg_50_list = []
    ndcg_d_list = []

    prec_10_list = []
    prec_50_list = []
    prec_all_list = []
    prec_20_list = []
    
    sig_list = []
    
    ndcg_10 = 0
    ndcg_50 = 0
    ndcg_d = 0

    import sklearn.metrics as metrics
    fpr, tpr, threshold = metrics.roc_curve(y_test, test_scores)
    ax_roc.plot(fpr, tpr,label=emb)
    
    precision, recall, threshold = metrics.precision_recall_curve(y_test, test_scores)
    ax_pr.plot(recall, precision, label=emb)
    #PrecisionRecallDisplay.from_predictions(y_test[:,0], test_scores, name=emb,ax=ax_pr)

    for v in sample_verts:
        if eval_all:
          edges = label_counts[v]
          adj = whole_adj
        else:
          edges = np.sum(test_adj[v,:])
          adj = test_adj

        if edges > 0:
            pairs = np.array([[v, i] for i in range(whole_adj.shape[0])], dtype=np.int32)

            #vec_feats = np.concatenate(([vecs[v] for i in range(whole_adj.shape[0])], vecs), axis=1)
            vec_feats = np.multiply(vecs[v], vecs)

            vec_row_scores = clf.predict_proba(vec_feats)[:,1]
            
            if not eval_all and not multigraph:
              for pair in pairs:
                  if train_adj[pair[0], pair[1]]:
                      vec_row_scores[pair[1]] = -np.inf
            
            vec_predictions = np.argpartition(-vec_row_scores, 100)

            prec_10 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:10]]]) / 10
            recall_10 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:10]]]) / edges
           
            prec_20 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:20]]]) / 20
            recall_20 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:20]]]) / edges
             
            prec_50 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:50]]]) / 50
            recall_50 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:50]]]) / edges
            
            prec_all = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:edges]]]) / edges
            recall_all = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:edges]]]) / edges
            
            dcg_10 = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:10]])])
            dcg_50 = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:50]])]) 
            dcg_d = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:label_counts[v]]])])
             
            idcg_10 = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(10)])])
            idcg_50 = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(50)])])
            idcg_d = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(edges)])])

            ndcg_10_list.append(dcg_10 / idcg_10)
            ndcg_50_list.append(dcg_50 / idcg_50)
            ndcg_d_list.append(dcg_d  / idcg_d)

            prec_10_list.append(max(prec_10, recall_10))
            prec_20_list.append(max(prec_20, recall_20))
 
            prec_50_list.append(max(prec_50, recall_50))
            prec_all_list.append(max(prec_all, recall_all))

            vec_precision_10  += prec_10_list[-1] 
            vec_precision_20  += prec_20_list[-1] 
            vec_precision_50  += prec_50_list[-1] 
            vec_precision_d += prec_all_list[-1] 
            
            ndcg_10 += ndcg_10_list[-1] 
            ndcg_50 += ndcg_50_list[-1] 
            ndcg_d += ndcg_d_list[-1] 

            # Code to plot top dot products
            #mean_neighbor_dot = np.mean([np.dot(vecs[v], vecs[i]) for i in g.neighbors(v)])

            #top_dot = []
            #offset = 0
            #while len(top_dot) < 10:
            #    pair = pairs[vec_predictions[len(top_dot)+offset]]
            #    if adj[v, pair[1]]:
            #        offset+=1
            #    else:
            #        top_dot.append(np.dot(vecs[pair[0]], vecs[pair[1]]))
            #mean_top_dot = np.mean(top_dot)
            #sig_list.append(mean_neighbor_dot/mean_top_dot)
            

    
    ax_rel_10.plot(np.arange(1000) / 1000, histofy(prec_10_list), label=emb)
    ax_rel_20.plot(np.arange(1000) / 1000, histofy(prec_20_list), label=emb)
    ax_rel_50.plot(np.arange(1000) / 1000, histofy(prec_50_list), label=emb)
    ax_rel_all.plot(np.arange(1000) / 1000, histofy(prec_all_list), label=emb)
    
    
    ax_ndcg_10.plot(np.arange(1000) / 1000, histofy(ndcg_10_list), label=emb)
    ax_ndcg_50.plot(np.arange(1000) / 1000, histofy(ndcg_50_list), label=emb)
    ax_ndcg_d.plot(np.arange(1000) / 1000, histofy(ndcg_d_list), label=emb)
    
    print("Hits@10 Score: %f" % (vec_precision_10/len(sample_verts)))
    print("Hits@20 Score: %f" % (vec_precision_20/len(sample_verts)))
    print("Hits@50 Score: %f" % (vec_precision_50/len(sample_verts)))
    print("Hits@d Score: %f" % (vec_precision_d/len(sample_verts)))        
    
    print("NDCG@10 Score: %f" % (np.average(ndcg_10_list)))
    print("NDCG@50 Score: %f" % (np.average(ndcg_50_list)))
    print("NCDG@d Score: %f" % (np.average(ndcg_d_list)))
    
    
    
    test_scores = clf.predict(X_test)
    print("Accuracy Score: %f" % accuracy_score(y_test, test_scores))
    
    #fig_hist, ax_hist = plt.subplots()
    #ax_hist.hist(sig_list, bins=50)
    #ax_hist.axvline(1, color='red')
    #from sklearn.preprocessing import normalize
    
    #ax_hist.set_title(f'Significance for {emb}') 
    #ax_hist.set_ylabel('Counts')
    #ax_hist.set_xlabel('Significance')
    #fig_hist.savefig(f'{dataset}-{emb}-significance.eps', format='eps', dpi=1200)
    return

def resource_allocation(A, edge_pair, batch_size=100000):
  w = 1 / A.sum(axis=0)
  w[np.isinf(w)] = 0
  D = A.multiply(w).tocsr()  # e[i,j] / log(d_j)
  edge_index = edge_pair.t()
  link_loader = DataLoader(range(edge_index.size(1)), batch_size)
  scores = []
  for idx in link_loader:
      src, dst = edge_index[0, idx], edge_index[1, idx]
      batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
      scores.append(batch_scores)
  scores = np.concatenate(scores, 0)
  return torch.FloatTensor(scores)

def CN(A, edge_pair, batch_size=100000):
    edge_index = edge_pair.t()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0))

def AA(A, edge_pair, batch_size=100000):
    edge_index = edge_pair.t()
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def simple_link_prediction(test_pair, y_test, whole_adj, label_counts, emb, sample_verts, g, train_adj, test_adj, hits_test_pair, hits_neg, evaluator, method, eval_all, multigraph):
    num_pos = np.count_nonzero(y_test)
    num_neg = np.sum(len(y_test) - np.count_nonzero(y_test))
    if method == 'ra':
      y_pos_scores = resource_allocation(train_adj, torch.from_numpy(test_pair[:num_pos]))
      y_neg_scores = resource_allocation(train_adj, torch.from_numpy(test_pair[num_neg:]))
    elif method == 'aa':
      y_pos_scores = AA(train_adj, torch.from_numpy(test_pair[:num_pos]))
      y_neg_scores = AA(train_adj, torch.from_numpy(test_pair[num_neg:]))
    else:
      y_pos_scores = CN(train_adj, torch.from_numpy(test_pair[:num_pos]))
      y_neg_scores = CN(train_adj, torch.from_numpy(test_pair[num_neg:]))

    test_scores = np.vstack((y_pos_scores.numpy().reshape(-1,1), y_neg_scores.numpy().reshape(-1,1)))
    print("ROC AUC SCORE: %f" % roc_auc_score(y_test, test_scores))
    p, r, t = precision_recall_curve(y_test, test_scores)
    print("PR AUC SCORE: %f" % auc(r, p))
    
    if method == 'ra':
      y_pos_scores = resource_allocation(train_adj, torch.from_numpy(hits_test_pair))
      y_neg_scores = resource_allocation(train_adj, torch.from_numpy(hits_neg))
    elif method == 'aa':
      y_pos_scores = AA(train_adj, torch.from_numpy(hits_test_pair))
      y_neg_scores = AA(train_adj, torch.from_numpy(hits_neg))
    else:
      y_pos_scores = CN(train_adj, torch.from_numpy(hits_test_pair))
      y_neg_scores = CN(train_adj, torch.from_numpy(hits_neg))

    result = evaluator.eval({'y_pred_pos': y_pos_scores, 'y_pred_neg': y_neg_scores})
    print(f'OGB Mean Hits@{args.hits_k} Score: {result[f"hits@{args.hits_k}"]}')

    vec_precision_10 = 0
    vec_precision_20 = 0  
    vec_precision_50 = 0
    vec_precision_d = 0
    
    ndcg_10_list = []
    ndcg_50_list = []
    ndcg_d_list = []

    prec_10_list = []
    prec_50_list = []
    prec_all_list = []
    prec_20_list = []
    
    sig_list = []
    
    ndcg_10 = 0
    ndcg_50 = 0
    ndcg_d = 0

    import sklearn.metrics as metrics
    fpr, tpr, threshold = metrics.roc_curve(y_test, test_scores)
    ax_roc.plot(fpr, tpr,label=emb)
    
    precision, recall, threshold = metrics.precision_recall_curve(y_test, test_scores)
    ax_pr.plot(recall, precision,label=emb)

    #PrecisionRecallDisplay.from_predictions(y_test[:,0], test_scores, name=emb,ax=ax_pr)
    
    for v in sample_verts:
        if eval_all:
          edges = label_counts[v]
          adj = whole_adj
        else:
          edges = np.sum(test_adj[v,:])
          adj = test_adj

        if edges > 0:
            pairs = np.array([[v, i] for i in range(whole_adj.shape[0])], dtype=np.int32)

            #vec_feats = np.multiply(vecs[v], vecs)
            
            if method == 'ra':
              row_scores = resource_allocation(train_adj, torch.from_numpy(pairs)).numpy()
            elif method == 'aa':
              row_scores = AA(train_adj, torch.from_numpy(pairs)).numpy()
            else:
              row_scores = CN(train_adj, torch.from_numpy(pairs)).numpy()
            
            if not eval_all and not multigraph:
              for pair in pairs:
                  if train_adj[pair[0], pair[1]]:
                      row_scores[pair[1]] = -np.inf
            
            vec_predictions = np.argpartition(-row_scores, 100)

            # TODO: Make more Efficient
            prec_10 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:10]]]) / 10
            recall_10 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:10]]]) / edges
           
            prec_20 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:20]]]) / 20
            recall_20 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:20]]]) / edges
             
            prec_50 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:50]]]) / 50
            recall_50 = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:50]]]) / edges
            
            prec_all = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:edges]]]) / edges
            recall_all = np.sum([adj[pair[0], pair[1]]  for pair in pairs[vec_predictions[:edges]]]) / edges
            
            dcg_10 = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:10]])])
            dcg_50 = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:50]])]) 
            dcg_d = np.sum([(adj[pair[0], pair[1]]/np.log2(2+idx)) for idx, pair in enumerate(pairs[vec_predictions[:label_counts[v]]])])
             
            idcg_10 = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(10)])])
            idcg_50 = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(50)])])
            idcg_d = np.sum([(rank/np.log2(2+idx)) for idx, rank in enumerate([1 if i < edges else 0 for i in range(edges)])])

            ndcg_10_list.append(dcg_10 / idcg_10)
            ndcg_50_list.append(dcg_50 / idcg_50)
            ndcg_d_list.append(dcg_d  / idcg_d)

            prec_10_list.append(max(prec_10, recall_10))
            prec_20_list.append(max(prec_20, recall_20))
 
            prec_50_list.append(max(prec_50, recall_50))
            prec_all_list.append(max(prec_all, recall_all))

            vec_precision_10 += prec_10_list[-1] 
            vec_precision_20 += prec_20_list[-1] 
            vec_precision_50 += prec_50_list[-1] 
            vec_precision_d += prec_all_list[-1] 
            
            ndcg_10 += ndcg_10_list[-1] 
            ndcg_50 += ndcg_50_list[-1] 
            ndcg_d += ndcg_d_list[-1] 

            # Code to plot top dot products
            #mean_neighbor_dot = np.mean([np.dot(vecs[v], vecs[i]) for i in g.neighbors(v)])

            #top_dot = []
            #offset = 0
            #while len(top_dot) < 10:
            #    pair = pairs[vec_predictions[len(top_dot)+offset]]
            #    if adj[v, pair[1]]:
            #        offset+=1
            #    else:
            #        top_dot.append(np.dot(vecs[pair[0]], vecs[pair[1]]))
            #mean_top_dot = np.mean(top_dot)
            #sig_list.append(mean_neighbor_dot/mean_top_dot)
            

    
    ax_rel_10.plot(np.arange(1000) / 1000, histofy(prec_10_list), label=emb)
    ax_rel_20.plot(np.arange(1000) / 1000, histofy(prec_20_list), label=emb)
    ax_rel_50.plot(np.arange(1000) / 1000, histofy(prec_50_list), label=emb)
    ax_rel_all.plot(np.arange(1000) / 1000, histofy(prec_all_list), label=emb)
    
    
    ax_ndcg_10.plot(np.arange(1000) / 1000, histofy(ndcg_10_list), label=emb)
    ax_ndcg_50.plot(np.arange(1000) / 1000, histofy(ndcg_50_list), label=emb)
    ax_ndcg_d.plot(np.arange(1000) / 1000, histofy(ndcg_d_list), label=emb)
    
    print("Hits@10 Score: %f" % (vec_precision_10/len(sample_verts)))
    print("Hits@20 Score: %f" % (vec_precision_20/len(sample_verts)))
    print("Hits@50 Score: %f" % (vec_precision_50/len(sample_verts)))
    print("Hits@d Score: %f" % (vec_precision_d/len(sample_verts)))        
    
    print("NDCG@10 Score: %f" % (np.average(ndcg_10_list)))
    print("NDCG@50 Score: %f" % (np.average(ndcg_50_list)))
    print("NCDG@d Score: %f" % (np.average(ndcg_d_list)))
    
        
    #fig_hist, ax_hist = plt.subplots()
    #ax_hist.hist(sig_list, bins=50)
    #ax_hist.axvline(1, color='red')
    #from sklearn.preprocessing import normalize
    
    #ax_hist.set_title(f'Significance for {emb}') 
    #ax_hist.set_ylabel('Counts')
    #ax_hist.set_xlabel('Significance')
    #fig_hist.savefig(f'{dataset}-{emb}-significance.eps', format='eps', dpi=1200)
    return


if args.sbm:
  g = nx.read_adjlist(f"{dataset}.adjlist", nodetype=int)
elif args.ogb:
  g = nx.read_edgelist(f"{dataset}.edgelist", nodetype=int) 
elif args.ppi:
    data = open(args.source_file,'r')
    g = nx.Graph()
    e_list = []
    for line in data:
        edges = line.split()
        for i in range(0,len(edges)//3):
            (u,v) = int(edges[3*i]),int(edges[(3*i)+1])
            e_list.append((u,v))
    g.add_edges_from(e_list)
    g = nx.convert_node_labels_to_integers(g)  
else:
  g = nx.read_edgelist(args.source_file)
  g = nx.convert_node_labels_to_integers(g)
whole_adj = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes()))


print(g.number_of_nodes())
print(g.number_of_edges())
train_ind = np.load(f"{dataset}-train-ind.npy")
test_ind = np.load(f"{dataset}-test-ind.npy")
train_pair = np.load(f'{dataset}-train-pair.npy')
test_pair = np.load(f'{dataset}-test-pair.npy')

train_g = copy.deepcopy(nx.Graph(g))
train_g.remove_edges_from(test_pair)
train_g.add_edges_from(train_pair)

test_g = copy.deepcopy(nx.Graph(g))
test_g.remove_edges_from(train_pair)
test_g.add_edges_from(test_pair)

test_adj = nx.to_scipy_sparse_matrix(test_g, nodelist=sorted(g.nodes()))
train_adj = nx.to_scipy_sparse_matrix(train_g, nodelist=sorted(g.nodes()))

print("Loaded Graph")
label_counts = np.asarray(np.sum(whole_adj, axis=1)).reshape(-1)
hits_test_pair = test_pair

print("Loaded Split")

rng = np.random.default_rng()
sample_verts = rng.choice(test_pair, size=1000, axis=0)[:, 0].flatten()
neg_train = rng.integers(low=0, high=whole_adj.shape[0], size=(len(train_ind), 2))
train_pair = np.vstack((train_pair, neg_train))
y_train = np.zeros( (2*len(train_ind), 1), dtype=bool)
y_train[:len(train_ind)] = True
neg_test = rng.integers(low=0, high=whole_adj.shape[0], size=(len(test_ind), 2))
test_pair = np.vstack((test_pair, neg_test))
y_test = np.zeros( (2*len(test_ind), 1), dtype=bool)
y_test[:len(test_ind)] = True
if args.ogb:
  data = LinkPropPredDataset(name = 'ogbl-collab')
  split_edge = data.get_edge_split()
  train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
  # hits_neg = test_edge['edge_neg'] 
hits_neg = rng.integers(low=0, high=whole_adj.shape[0], size=(g.number_of_edges()//10, 2))
hits_neg = np.asarray([neg_pair for neg_pair in hits_neg if whole_adj[neg_pair[0], neg_pair[1]]==0 ])
print("Processed Split")

if args.hits_k == 20:
  evaluator = Evaluator(name = f'ogbl-ddi')
else:
  evaluator = Evaluator(name = f'ogbl-collab')

for emb in embedding_names:
    print(emb)
    if emb in ['ra', 'aa', 'cn']:
      simple_link_prediction(test_pair, y_test, whole_adj, label_counts, emb, sample_verts, g, train_adj, test_adj, hits_test_pair, hits_neg, evaluator, emb, eval_all=False, multigraph=args.Multigraph)
    else:
      if emb in ['graphsage-pool', 'graphsage-mean', 'graphsage-lstm']:
          vecs = np.load(f'{emb}-{dataset}-val.npy')
          shuf_rows = {}
          with open(f'{emb}-{dataset}-val.txt') as f:
              for i, line in enumerate(f): 
                  shuf_rows[int(line.strip())] = i
          vecs = vecs[[shuf_rows[id] for id in range(vecs.shape[0])]]
      elif emb in ['hoprec','hpe']:
        with open(f'{dataset}-{emb}', 'r') as f:
          for i, line in enumerate(f):
            if i == 0:   
                dim = int(line.rstrip('\n').split(' ')[1])
                vecs = np.zeros((whole_adj.shape[0],dim))
            else:  
                entity_embed = line.rstrip('\n').split(' ')
                vecs[int(entity_embed[0])] = np.array(entity_embed[1:], dtype=float)
      else:
          vecs = np.load(f'{emb}-{dataset}.npy')
      if args.Eval_all_edges:
        link_prediction(vecs, train_pair, test_pair, y_train, y_test, whole_adj, label_counts, emb, sample_verts, g, train_adj, test_adj, hits_test_pair, hits_neg, eval_all=True, multigraph=args.Multigraph)
      else:
        link_prediction(vecs, train_pair, test_pair, y_train, y_test, whole_adj, label_counts, emb, sample_verts, g, train_adj, test_adj, hits_test_pair, hits_neg, eval_all=False, multigraph=args.Multigraph)

ax_rel_10.set_title('Distribution of VCMPR@10 scores')
ax_rel_10.set_ylabel('Percent of nodes')
ax_rel_10.set_xlabel('MaxPR@10')
ax_rel_10.legend()
fig_rel_10.savefig(f'{dataset}-vcmpr-plot-10.eps', format='eps', dpi=1200)

ax_rel_50.set_title('Distribution of VCMPR@50 scores')
ax_rel_50.set_ylabel('Percent of nodes')
ax_rel_50.legend()
ax_rel_50.set_xlabel('MaxPR@50')
fig_rel_50.savefig(f'{dataset}-vcmpr-plot-50.eps', format='eps', dpi=1200)

ax_rel_all.set_title('Distribution of VCMPR@deg(v) scores')
ax_rel_all.set_ylabel('Percent of nodes')
ax_rel_all.set_xlabel('MaxPR@deg(v)')
ax_rel_all.legend()
fig_rel_all.savefig(f'{dataset}-vcmpr-plot-d.eps', format='eps', dpi=1200)

ax_rel_20.set_title('Distribution of VCMPR@20 scores')
ax_rel_20.set_ylabel('Percent of nodes')
ax_rel_20.set_xlabel('MaxPR@20')
ax_rel_20.legend()
fig_rel_20.savefig(f'{dataset}-vcmpr-plot-20.eps', format='eps', dpi=1200)

ax_ndcg_10.set_title('Distribution of VCNDCG@10 scores')
ax_ndcg_10.set_ylabel('Percent of nodes')
ax_ndcg_10.set_xlabel('VCNDCG@10')
ax_ndcg_10.legend()
fig_ndcg_10.savefig(f'{dataset}-vcndcg-plot-10.eps', format='eps', dpi=1200)

ax_ndcg_50.set_title('Distribution of VCNDCG@50 scores')
ax_ndcg_50.set_ylabel('Percent of nodes')
ax_ndcg_50.set_xlabel('VCNDCG@50')
ax_ndcg_50.legend()
fig_ndcg_50.savefig(f'{dataset}-ndcg-plot-50.eps', format='eps', dpi=1200)

ax_ndcg_d.set_title('Distribution of VCNDCG@deg(v) scores')
ax_ndcg_d.set_ylabel('Percent of nodes')
ax_ndcg_d.set_xlabel('VCNDCG@deg(v)')
ax_ndcg_d.legend()
fig_ndcg_d.savefig(f'{dataset}-vcndcg-plot-d.eps', format='eps', dpi=1200)

ax_pr.set_title('PR Curve')
ax_pr.set_ylabel('Precision')
ax_pr.set_xlabel('Recall')
ax_pr.legend()
fig_pr.savefig(f'{dataset}-PR-Curve.eps', format='eps', dpi=1200)

ax_roc.set_title('Roc Curve')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.legend()
fig_roc.savefig(f'{dataset}-ROC-Curve.eps', format='eps', dpi=1200)
