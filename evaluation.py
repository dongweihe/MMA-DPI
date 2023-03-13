from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import paddle
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
import pandas as pd
from operator import itemgetter
from argparse import ArgumentParser

from HGCNModule.deep.optimizer import HGCNOptimizer
from HGCNModule.deep.model import HGCNModel
from HGCNModule.deep.minibatch import EdgeMinibatchIterator
from HGCNModule.utility import rank_metrics, preprocessing
from HGCNModule.utility import loadData
from TransModule.helper import utils
from TransModule.double_towers import MolTransModel
from TransModule.preprocess import DataEncoder

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, auc)

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


paddle.seed(2)
np.random.seed(3)

# Whether to use GPU
# USE_GPU = True
# Set device as $export CUDA_VISIBLE_DEVICES='your device number'
use_cuda = paddle.is_compiled_with_cuda()
device = 'cuda:0' if use_cuda else 'cpu'
device = device.replace('cuda', 'gpu')
device = paddle.set_device(device)

def get_mol_att_db(db_name):
    """
    Get benchmark dataset for classification
    """
    if db_name.lower() == 'davis':
        return 'data/micro attribute/DAVIS'
    elif db_name.lower() == 'bindingdb':
        return 'data/micro attribute/BindingDB'

def get_accuracy_scores(minibatch,sess, opt, adj_mats_orig,feed_dict, edges_pos, edges_neg, edge_type):
    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    placeholders = construct_placeholders(edge_types)
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0

    # pos
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []

    # neg
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    # evatalution.....
    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=10)

    return roc_sc, aupr_sc, apk_sc


def get_final_accuracy_scores(minibatch,sess, opt, adj_mats_orig,feed_dict, edges_pos, edges_neg, edge_type):
    edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
    placeholders = construct_placeholders(edge_types)
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0

    # pos
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    # neg
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    # evatalution.....
    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)
    FPR, TPR, thresholds = metrics.roc_curve(labels_all, preds_all)

    precision, recall, _ = metrics.precision_recall_curve(labels_all, preds_all)

    mse = metrics.mean_squared_error(labels_all, preds_all)
    mae = metrics.median_absolute_error(labels_all, preds_all)
    r2 = metrics.r2_score(labels_all, preds_all)
    preds_all[preds_all >= 0.5] = 1
    preds_all[preds_all < 0.5] = 0
    acc = metrics.accuracy_score(labels_all, preds_all)
    f1 = metrics.f1_score(labels_all, preds_all, average='macro')
    return FPR, TPR, roc_sc, \
           precision, recall, aupr_sc, \
           apk_sc, thresholds, mse, mae, r2, acc, f1


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int64, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int64, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int64),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types})

    return placeholders


def test(sig, loss_fn,data_generator, model):
    y_pred = []
    y_label = []
    loss_res = 0.0
    count = 0.0

    model.eval()
    for _, data in enumerate(data_generator):
        d_out, mask_d_out, t_out, mask_t_out, label = data
        temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
        predicts = paddle.squeeze(sig(temp))
        label = paddle.cast(label, "float32")

        loss = loss_fn(predicts, label)
        loss_res += loss
        count += 1

        predicts = predicts.detach().cpu().numpy()
        label_id = label.to('cpu').numpy()
        y_label = y_label + label_id.flatten().tolist()
        y_pred = y_pred + predicts.flatten().tolist()
    loss = loss_res / count

    fpr, tpr, threshold = roc_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 1e-05)
    optimal_threshold = threshold[5:][np.argmax(f1[5:])]
    print("Optimal threshold: {}".format(optimal_threshold))

    y_pred_res = [(1 if i else 0) for i in y_pred >= optimal_threshold]
    auroc = auc(fpr, tpr)
    print("AUROC: {}".format(auroc))
    print("AUPRC: {}".format(average_precision_score(y_label, y_pred)))

    cf_mat = confusion_matrix(y_label, y_pred_res)
    print("Confusion Matrix: \n{}".format(cf_mat))
    print("Precision: {}".format(precision_score(y_label, y_pred_res)))
    print("Recall: {}".format(recall_score(y_label, y_pred_res)))

    total_res = sum(sum(cf_mat))
    accuracy = (cf_mat[0, 0] + cf_mat[1, 1]) / total_res
    print("Accuracy: {}".format(accuracy))
    sensitivity = cf_mat[0, 0] / (cf_mat[0, 0] + cf_mat[0, 1])
    print("Sensitivity: {}".format(sensitivity))
    specificity = cf_mat[1, 1] / (cf_mat[1, 0] + cf_mat[1, 1])
    print("Specificity: {}".format(specificity))
    outputs = np.asarray([(1 if i else 0) for i in np.asarray(y_pred) >= 0.5])
    return (roc_auc_score(y_label, y_pred),
            f1_score(y_label, outputs), loss.item())

# Set loss function
def run_MMADTI():
    sig = paddle.nn.Sigmoid()
    loss_fn = paddle.nn.BCELoss() #binary cross entropy


    subdata_Net = 'Luo'


    parser = ArgumentParser(description='Start Training...')
    parser.add_argument('-b', '--batchsize', default=128, type=int, metavar='N', help='Batch size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='Number of workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--dataset', choices=['davis', 'bindingdb'], default='bindingdb',
                        type=str, metavar='DATASET', help='Select specific dataset for your task')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./Trans_config.json', type=str)
    args = parser.parse_args()



    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
    flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1',64, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
    flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
    flags.DEFINE_boolean('bias', True, 'Bias term.')

    Trans_config = {
        "drug_max_seq": 50,
        "target_max_seq": 545,
        "emb_size": 384,
        "input_drug_dim": 23532,
        "input_target_dim": 16693,
        "interm_size": 1536,
        "num_attention_heads": 12,
        "flatten_dim": 81750,
        "layer_size": 2,
        "dropout_ratio": 0.1,
        "attention_dropout_ratio": 0.1,
        "hidden_dropout_ratio": 0.1
    }



    #01
    AUROC_01_list = []
    AUPR_01_list  = []
    APatK_01_list = []
    ACC_01_list   = []
    F1_01_list    = []
    MSE_01_list   = []
    MAE_01_list   = []

    #10
    AUROC_10_list = []
    AUPR_10_list  = []
    APatK_10_list = []
    ACC_10_list   = []
    F1_10_list    = []
    MSE_10_list   = []
    MAE_10_list   = []

    # About drug
    drug_drug_path = 'data/macro attribute/Luo/sevenNets/mat_drug_drug.txt'
    drug_drug_sim_chemical_path = 'data/macro attribute/Luo/sim_network/Sim_mat_Drugs.txt'
    drug_drug_sim_interaction_path = 'data/macro attribute/Luo/sim_network/Sim_mat_drug_drug.txt'
    drug_drug_sim_se_path = 'data/macro attribute/Luo/sim_network/Sim_mat_drug_se.txt'
    drug_drug_sim_disease_path = 'data/macro attribute/Luo/sim_network/Sim_mat_drug_disease.txt'
    drug_protein_path = 'data/macro attribute/Luo/sevenNets/mat_drug_protein.txt'

    # About Protein
    protein_drug_path = 'data/macro attribute/Luo/sevenNets/mat_protein_drug.txt'
    protein_protein_path = 'data/macro attribute/Luo/sevenNets/mat_protein_protein.txt'
    protein_protein_sim_sequence_path = 'data/macro attribute/Luo/sim_network/Sim_mat_Proteins.txt'
    protein_protein_sim_disease_path = 'data/macro attribute/Luo/sim_network/Sim_mat_protein_disease.txt'
    protein_protein_sim_interaction_path = 'data/macro attribute/Luo/sim_network/Sim_mat_protein_protein.txt'

    # About drug and protein (others)...
    protein_disease_path = 'data/macro attribute/Luo/sevenNets/mat_protein_disease.txt'
    drug_disease_path = 'data/macro attribute/Luo/sevenNets/mat_drug_disease.txt'
    drug_sideEffect_path = 'data/macro attribute/Luo/sevenNets/mat_drug_se.txt'


    # Step1:Construct the graph(read the data...)
    print("HGCN Moudle data loading")
    # drug_drug_adj and protein_protein_adj combine the simNets and interactions
    drug_drug_adj = loadData.Load_Drug_Adj_Togerther(drug_drug_path=drug_drug_path,
                                                     drug_drug_sim_chemical_path=drug_drug_sim_chemical_path,
                                                     drug_drug_sim_interaction_path=drug_drug_sim_interaction_path,
                                                     drug_drug_sim_se_path=drug_drug_sim_se_path,
                                                     drug_drug_sim_disease_path=drug_drug_sim_disease_path)

    protein_protein_adj = loadData.Load_Protein_Adj_Togerther(protein_protein_path=protein_protein_path,
                                                              protein_protein_sim_sequence_path=protein_protein_sim_sequence_path,
                                                              protein_protein_sim_disease_path=protein_protein_sim_disease_path,
                                                              protein_protein_sim_interaction_path=protein_protein_sim_interaction_path)

    drug_proten_interactions, protein_drug_interactions = loadData.load_protein_drug_interactions(path=protein_drug_path)

    protein_disease_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=protein_disease_path)
    disease_protein_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=protein_disease_path)
    drug_disease_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=drug_disease_path)
    disease_drug_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=drug_disease_path)

    drug_side_effect_adj = loadData.load_Adj_adj(threshold=0, toone=0, draw=0, sim_path=drug_sideEffect_path)
    side_effect_drug_adj = loadData.load_Adj_adj_transpose(threshold=0, toone=0, draw=0, sim_path=drug_sideEffect_path)

    optimal_auc = 0
    log_iter = 50  # 一个epoch迭代多少次
    log_step = 0
    et_opreation = 'FCNNOpreation'

    # Load model config
    # model_config = json.load(open(args.model_config, 'r'))
    MTModule = MolTransModel(Trans_config)
    MTModule = MTModule.cuda()

    # Load pretrained model
    # params_dict= paddle.load('./pretrained_model/pdb2016_single_tower_1')
    # model.set_dict(params_dict)

    # Optimizer
    # scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=args.lr, warmup_steps=50, start_lr=0,
    #                                             end_lr=args.lr, verbose=False)
    optim = utils.Adam(parameters=MTModule.parameters(), learning_rate=args.lr)  # Adam
    # optim = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=0.01) # AdamW

    # Data Preparation
    data_path = get_mol_att_db(args.dataset)
    training_set = pd.read_csv(data_path + '/train.csv')  # bindingDB 6334 DPI
    validation_set = pd.read_csv(data_path + '/val.csv')
    testing_set = pd.read_csv(data_path + '/test.csv')

    print("Trans Moudle dataset loading")
    print("<><><><><><>Load Drug SIMLES, Protein Sequence, String Position and Labels<><><><><><>")
    training_data = DataEncoder(training_set.index.values, training_set.Label.values, training_set)
    train_loader = utils.BaseDataLoader(training_data, batch_size=args.batchsize, shuffle=True,
                                        drop_last=False, num_workers=args.workers)
    validation_data = DataEncoder(validation_set.index.values, validation_set.Label.values, validation_set)
    validation_loader = utils.BaseDataLoader(validation_data, batch_size=args.batchsize, shuffle=False,
                                             drop_last=False, num_workers=args.workers)
    testing_data = DataEncoder(testing_set.index.values, testing_set.Label.values, testing_set)
    testing_loader = utils.BaseDataLoader(testing_data, batch_size=args.batchsize, shuffle=False,
                                          drop_last=False, num_workers=args.workers)
    #Nested 10-fold CV
    for fold in range(0, 10):
        val_test_size = 0.1
        print('Current fold is :', fold)


        # data representation
        # 0 for protein / 1 for drug / 2 for disease / 3 for side-effect
        adj_mats_orig = {
            (0, 0): [protein_protein_adj, protein_protein_adj],  # type1
            (0, 1): [protein_drug_interactions],  # type2
            (0, 2): [protein_disease_adj],

            (1, 0): [drug_proten_interactions],
            (1, 1): [drug_drug_adj, drug_drug_adj],  # type3
            (1, 2): [drug_disease_adj],
            (1, 3): [drug_side_effect_adj],

            (2, 0): [disease_protein_adj],
            (2, 1): [disease_drug_adj],

            (3, 1): [side_effect_drug_adj],
        }

        protein_degrees = np.array(protein_protein_adj.sum(axis=0)).squeeze()
        drug_degrees = np.array(drug_drug_adj.sum(axis=0)).squeeze()
        disease_degrees = np.array(disease_drug_adj.sum(axis=0)).squeeze()
        side_effect_degrees = np.array(side_effect_drug_adj.sum(axis=0)).squeeze()

        degrees = {
            0: [protein_degrees, protein_degrees],
            1: [drug_degrees, drug_degrees],
            2: [disease_degrees],
            3: [side_effect_degrees]
        }

        # # featureless (genes)
        gene_feat = sp.identity(1512)
        protein_nonzero_feat, protein_num_feat = gene_feat.shape
        gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

        #
        # # features (drugs)
        drug_feat = sp.identity(708)
        # drug_feat = Drug_Drug_adj
        drug_nonzero_feat, drug_num_feat = drug_feat.shape
        drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

        # data representation
        diease_feat = sp.identity(5603)
        diease_nonzero_feat, diease_num_feat = diease_feat.shape
        diease_feat = preprocessing.sparse_to_tuple(diease_feat.tocoo())
        # NOTICE

        side_effect_feat = sp.identity(4192)
        side_effect_nonzero_feat, side_effect_num_feat = side_effect_feat.shape
        side_effect_feat = preprocessing.sparse_to_tuple(side_effect_feat.tocoo())
        extra_side_effect_feat = side_effect_feat
        # NOTICE

        num_feat = {
            0: protein_num_feat,
            1: drug_num_feat,
            2: diease_num_feat,
            3: side_effect_num_feat,
        }
        nonzero_feat = {
            0: protein_nonzero_feat,
            1: drug_nonzero_feat,
            2: diease_nonzero_feat,
            3: side_effect_nonzero_feat
        }
        feat = {
            0: gene_feat,
            1: drug_feat,
            2: diease_feat,
            3: side_effect_feat

        }

        edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}

        # edge_types
        # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
        edge_type2decoder = {
            (0, 0): et_opreation,
            (0, 1): et_opreation,
            (0, 2): et_opreation,

            (1, 0): et_opreation,
            (1, 1): et_opreation,
            (1, 2): et_opreation,
            (1, 3): et_opreation,

            (2, 0): et_opreation,
            (2, 1): et_opreation,

            (3, 1): et_opreation,
        }

        edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
        print('edge_types', edge_types)
        num_edge_types = sum(edge_types.values())
        print("Edge types:", "%d" % num_edge_types)

        num_iter = 20

        print("Defining placeholders")
        placeholders = construct_placeholders(edge_types)

        print("Create minibatch iterator")
        minibatch = EdgeMinibatchIterator(
            adj_mats=adj_mats_orig,
            feat=feat,
            seed=fold,
            data_set=subdata_Net,
            edge_types=edge_types,
            batch_size=FLAGS.batch_size,
            val_test_size=val_test_size
        )

        print("Create model")
        HGCNModule = HGCNModel(
            placeholders=placeholders,
            num_feat=num_feat,
            nonzero_feat=nonzero_feat,
            data_set=subdata_Net,
            edge_types=edge_types,
            decoders=edge_type2decoder,
        )

        print("Create optimizer")
        with tf.name_scope('optimizer'):
            opt = HGCNOptimizer(
                embeddings=HGCNModule.embeddings,
                latent_inters=HGCNModule.latent_inters,
                latent_varies=HGCNModule.latent_varies,
                degrees=degrees,
                edge_types=edge_types,
                edge_type2dim=edge_type2dim,
                placeholders=placeholders,
                batch_size=FLAGS.batch_size,
                margin=FLAGS.max_margin
            )

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        feed_dict = {}

        # Train model
        print("Train model")
        for epoch in range(FLAGS.epochs):
            minibatch.shuffle()
            itr = 0
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
                feed_dict = minibatch.update_feed_dict(feed_dict=feed_dict, dropout=FLAGS.dropout,
                                                       placeholders=placeholders)

                t = time.time()

                # Training step: run single weight update
                outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
                train_cost = outs[1]
                batch_edge_type = outs[2]

                if itr % num_iter == 0:
                    val_auc, val_auprc, val_apk = get_accuracy_scores(minibatch,sess, opt,adj_mats_orig,feed_dict,
                        minibatch.val_edges, minibatch.val_edges_false,
                        minibatch.idx2edge_type[minibatch.current_edge_type_idx])

                    print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                          "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

                itr += 1

            print("=====Start Initial Testing=====")
            with paddle.no_grad():  # 在计算图中不进行反向传播，节约内存资源
                auroc, f1, loss = test(sig, loss_fn,testing_loader, MTModule)
                print("Initial testing set: AUROC: {}, F1: {}, Testing loss: {}".format(auroc, f1, loss))

            MTModule.train()
            for batch_id, data in enumerate(train_loader):
                d_out, mask_d_out, t_out, mask_t_out, label = data
                # d_out 50长度的DRUG SIMLES，mask_d_out 50长度对应SIMLES有字符的地方为1，
                # t_out 545的PROTEIN Sequence，mask_d_out 545长度对应Sequence有字符的地方为1
                temp = MTModule(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda())
                label = paddle.cast(label, "float32")
                predicts = paddle.squeeze(sig(temp))
                loss = loss_fn(predicts, label)

                optim.clear_grad()
                loss.backward()
                optim.step()

                if batch_id % log_iter == 0:
                    print("Training at epoch: {}, step: {}, loss is: {}"
                          .format(epoch, batch_id, loss.cpu().detach().numpy()))
                    log_step += 1

                    # Validation
            print("=====Start Validation=====")
            with paddle.no_grad():
                auroc, f1, loss = test(sig, loss_fn,validation_loader, MTModule)
                print("Validation at epoch: {}, AUROC: {}, F1: {}, loss is: {}"
                      .format(epoch, auroc, f1, loss))

                # Save best model
                if auroc > optimal_auc:
                    optimal_auc = auroc
                    print("Saving the best_model...")
                    print("Best AUROC: {}".format(optimal_auc))
                    paddle.save(MTModule.state_dict(), 'bestAUC_model')

        print("AUROC: {}".format(optimal_auc))
        paddle.save(MTModule.state_dict(), 'final_model')

        # Load the trained model
        params_dict = paddle.load('bestAUC_model')
        MTModule.set_dict(params_dict)

        # Testing
        print("=====Start Testing=====")
        with paddle.no_grad():
            try:
                auroc, f1, loss = test(sig, loss_fn, testing_loader, MTModule)
                print("Testing result: AUROC: {}, F1: {}, Testing loss is: {}".format(auroc, f1, loss))
            except:
                print("Testing failed...")


        print("Optimization finished!")

        # et:edge type
        for et in range(num_edge_types):
            print('et=', et)
            PRINT = 1
            if PRINT == 1:
                FPR, TPR, roc_score, \
                precision, recall, auprc_score, \
                apk_score, \
                thresholds, mse, mae, r2, acc, f1 = get_final_accuracy_scores(minibatch, sess, opt, adj_mats_orig, feed_dict,
                    minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
                # if et==1 or et==2:
                # edge_types
                # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
                if et == 2:
                    AUROC_01_list.append(roc_score)
                    AUPR_01_list.append(auprc_score)
                    APatK_01_list.append(apk_score)
                    ACC_01_list.append(acc)
                    F1_01_list.append(f1)
                    MSE_01_list.append(mse)
                    MAE_01_list.append(mae)
                if et == 4:
                    AUROC_10_list.append(roc_score)
                    AUPR_10_list.append(auprc_score)
                    APatK_10_list.append(apk_score)
                    ACC_10_list.append(acc)
                    F1_10_list.append(f1)
                    MSE_10_list.append(mse)
                    MAE_10_list.append(mae)

                print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
                print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
                print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
                print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
                print("Edge type:", "%04d" % et, "Test acc score", "{:.5f}".format(acc))
                print("Edge type:", "%04d" % et, "Test f1 score", "{:.5f}".format(f1))
                print("Edge type:", "%04d" % et, "Test mse score", "{:.5f}".format(mse))
                print("Edge type:", "%04d" % et, "Test mae score", "{:.5f}".format(mae))
                print("Edge type:", "%04d" % et, "Test r2 score", "{:.5f}".format(r2))
                print()


    print('10-Flod-cross-val-result')

    print('-----10------')
    print('AUROC_10_list', AUROC_10_list)
    print('AUPR_10_list', AUPR_10_list)
    print('APatK_10_list', APatK_10_list)
    print('ACC_10_list', ACC_10_list)
    print('F1_10_list', F1_10_list)
    print('MSE_10_list', MSE_10_list)
    print('MAE_10_list', MAE_10_list)
    print('AVG_AUROC_10_list', np.mean(AUROC_10_list).round(4))
    print('AVG_AUPR_10_list', np.mean(AUPR_10_list).round(4))
    print('AVG_APatK_10_list', np.mean(APatK_10_list).round(4))
    print('AVG_ACC_10_list', np.mean(ACC_10_list).round(4))
    print('AVG_F1_10_list', np.mean(F1_10_list).round(4))
    print('AVG_MSE_10_list', np.mean(MSE_10_list).round(4))
    print('AVG_MAE_10_list', np.mean(MAE_10_list).round(4))

if __name__ == "__main__":
    run_MMADTI()