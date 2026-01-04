import argparse
import json
import logging
import os
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "1.5"
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning
from transformers import BertModel, BertTokenizer

from dataloader import tokenize_text, TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter

import copy
import pickle
from collections import defaultdict
from utils import *

query_name_dict = {('e',('r','o')): '1p-only',
                    ('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e',('r', 'r', 'o')): '2p-only',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'o')): '3p-only',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), 'o'): '2i-only',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), 'o'): '3i-only',
                    ('e', ('r', 'n')): '1n',
                    ('e', ('r', 'n', 'o')): '1n-only',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r', 'n')), 'o'): '2in-only',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')), 'o'): '3in-only',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                    (('e', ('r',)), ('e', ('r',)), ('u',), 'o'): '2u-only',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',)): '3u',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',), 'o'): '3u-only'
                }

query_only2query_structure = {
     ('e', ('r', 'o')) : ('e', ('r',)),
     ('e', ('r', 'r', 'o')) : ('e', ('r', 'r')),
     ('e', ('r', 'r', 'r', 'o')) : ('e', ('r', 'r', 'r')),
     (('e', ('r',)), ('e', ('r',)), 'o') : (('e', ('r',)), ('e', ('r',))),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
     ('e', ('r', 'n', 'o')) : ('e', ('r', 'n')),
     (('e', ('r',)), ('e', ('r', 'n')), 'o') : (('e', ('r',)), ('e', ('r', 'n'))),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
     (('e', ('r',)), ('e', ('r',)), ('u',), 'o') : (('e', ('r',)), ('e', ('r',)), ('u',)),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',))
     }

query_structure2query_only = {value: key for key, value in query_only2query_structure.items()}
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM', '2s', '3s', 'sp', 'us', 'is', 'ins', 'pnsi']


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_ablation_on_symbolic', action='store_true', help='do ablation')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('--data_directory', type=str, default='/workspace/HybridQA/Datasets', help="KG data path")
    parser.add_argument('--dataset', type=str, default='amazon')

    parser.add_argument('-d', '--hidden_dim', default=800, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--drop', type=float, default=0., help='dropout rate')
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=0, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")

    parser.add_argument('--graph_type', type=str, default='triple', help='DO NOT MANUALLY SET')
    ## amazon dataset
    parser.add_argument('--nprod', type=int, default=0, help='DO NOT MANUALLY SET') # num of products
    parser.add_argument('--nattr', type=int, default=0, help='DO NOT MANUALLY SET') # num of attributes
    parser.add_argument('--nattrval', type=int, default=0, help='DO NOT MANUALLY SET') # num of attribute values
    ## prime dataset
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET') 
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--save_checkpoint_steps', default=5000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--print_on_screen', action='store_true')
    parser.add_argument('--reg_weight', default=1e-3, type=float)
    parser.add_argument('--optimizer', choices=['adam', 'adagrad'], default='adam')
    parser.add_argument('--geo', default='beta', type=str, choices=['beta', 'gamma'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--use-qa-iterator', action='store_true', default=False)
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.1n.2in.3in.2u.3u', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-gammam', '--gamma_mode', default="(1600,2)", type=str,
                        help='(hidden_dim,num_layer) for GammaE relational projection')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
    parser.add_argument('-cenr', '--center_reg', default=0.02, type=float,
                        help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
    parser.add_argument('--params_frozen', action='store_true', help='freeze parameters')
    parser.add_argument('--graph_added', action='store_true', help='add graph information')

    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer, id2text_queries):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict, id2text_queries)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics

def load_ablation_data(args, tasks):
    
    args.ablation_data_path = os.path.join(args.data_directory, 'ablation', args.dataset)

    logging.info("loading ablation data")
    test_symbolic_queries = pickle.load(open(os.path.join(args.ablation_data_path, "test_symbolic_queries.pkl"), 'rb'))
    test_symbolic_query_hard_answers = pickle.load(open(os.path.join(args.ablation_data_path, "test_symbolic_query_hard_answers.pkl"), 'rb'))
    test_symbolic_query_easy_answers = pickle.load(open(os.path.join(args.ablation_data_path, "test_symbolic_query_easy_answers.pkl"), 'rb'))

    return test_symbolic_queries, test_symbolic_query_hard_answers, test_symbolic_query_easy_answers

def load_data(args, tasks):

    tasks = tasks.split('.')
    data_path = osp.join(args.data_directory, args.dataset)

    if args.dataset == 'amazon':
        train_queries = pickle.load(open(osp.join(data_path, 'train-queries.pkl'), 'rb'))
        train_answers = pickle.load(open(osp.join(data_path, 'train-answers.pkl'), 'rb'))
        valid_queries = pickle.load(open(osp.join(data_path, 'valid-queries.pkl'), 'rb'))
        valid_easy_answers = pickle.load(open(osp.join(data_path, 'valid-easy-answers.pkl'), 'rb'))
        valid_hard_answers = pickle.load(open(osp.join(data_path, 'valid-hard-answers.pkl'), 'rb'))
        #test_queries = pickle.load(open('/workspace/HybridQA/Datasets/ablation/amazon/percent_40.pkl', 'rb'))
        test_queries = pickle.load(open(osp.join(args.data_directory, 'samples', args.dataset, 'test-queries.pkl'), 'rb'))
        test_easy_answers = pickle.load(open(osp.join(data_path, 'test-easy-answers.pkl'), 'rb'))
        test_hard_answers = pickle.load(open(osp.join(data_path, 'test-hard-answers.pkl'), 'rb'))
        attr_id2ent = pickle.load(open(osp.join(data_path, 'attr_id2ent.pkl'), 'rb'))
        attr_id2rel = pickle.load(open(osp.join(data_path, 'attr_id2rel.pkl'), 'rb'))
        prod_id2ent = pickle.load(open(osp.join(data_path, 'prod_id2ent.pkl'), 'rb'))
        prodid2title = pickle.load(open(osp.join(data_path, 'prodid2title.pkl'), 'rb'))

        # remove tasks not in args.tasks
        for name in all_tasks:
            if name not in tasks:
                query_structure = name_query_dict[name]
                if query_structure in train_queries:
                    del train_queries[query_structure]
                if query_structure in valid_queries:
                    del valid_queries[query_structure]
                if query_structure in test_queries:
                    del test_queries[query_structure]

        if args.graph_added:
            queries_1p = pickle.load(open(osp.join(data_path, '1p-queries.pkl'), 'rb'))
            answers_1p = pickle.load(open(osp.join(data_path, '1p-fn-answers.pkl'), 'rb'))

            for query in queries_1p[('e', ('r',))]:
                train_queries[('e', ('r',))].add((query,''))  
                train_answers[(query,'')] = answers_1p[query]

            ## incorporate graph queries into train_queries
            #train_queries[('e', ('r','o'))] = set()
            #for query in queries_1p[('e', ('r',))]:
            #    train_queries[('e', ('r','o'))].add((query,''))  
            #    train_answers[(query,'')] = answers_1p[query] # ((117675, (4,)), '')

        return train_queries, train_answers, valid_queries, valid_easy_answers, valid_hard_answers, \
            test_queries, test_easy_answers, test_hard_answers, \
            attr_id2ent, attr_id2rel, prod_id2ent, prodid2title

    elif args.dataset == 'prime':
        train_queries = pickle.load(open(osp.join(data_path, 'train-queries.pkl'), 'rb'))
        train_answers = pickle.load(open(osp.join(data_path, 'train-answers.pkl'), 'rb'))
        valid_queries = pickle.load(open(osp.join(data_path, 'valid-queries.pkl'), 'rb'))
        valid_easy_answers = pickle.load(open(osp.join(data_path, 'valid-easy-answers.pkl'), 'rb'))
        valid_hard_answers = pickle.load(open(osp.join(data_path, 'valid-hard-answers.pkl'), 'rb'))
        test_queries = pickle.load(open(osp.join(args.data_directory, 'samples', args.dataset, 'test-queries.pkl'), 'rb'))
        test_easy_answers = pickle.load(open(osp.join(data_path, 'test-easy-answers.pkl'), 'rb'))
        test_hard_answers = pickle.load(open(osp.join(data_path, 'test-hard-answers.pkl'), 'rb'))
        id2ent = pickle.load(open(osp.join(data_path, 'id2ent.pkl'), 'rb'))
        id2rel = pickle.load(open(osp.join(data_path, 'id2rel.pkl'), 'rb'))
        ent_id2title = pickle.load(open(osp.join(data_path, 'ent_id2title.pkl'), 'rb'))

        # remove tasks
        for name in all_tasks:
            if name not in tasks:
                query_structure = name_query_dict[name]
                if query_structure in train_queries:
                    del train_queries[query_structure]
                if query_structure in valid_queries:
                    del valid_queries[query_structure]
                if query_structure in test_queries:
                    del test_queries[query_structure]

        if args.graph_added:
            queries_1p = pickle.load(open(osp.join(data_path, '1p-queries.pkl'), 'rb'))
            answers_1p = pickle.load(open(osp.join(data_path, '1p-fn-answers.pkl'), 'rb'))

            ## incorporate graph queries into train_queries
            for query in queries_1p[('e', ('r',))]:
                train_queries[('e', ('r',))].add((query,'')) 
                train_answers[(query,'')] = answers_1p[query]
        
        return test_queries, test_easy_answers, test_hard_answers, \
            id2ent, id2rel, ent_id2title, train_queries, train_answers, valid_queries, valid_easy_answers, valid_hard_answers, \

def get_bert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # Use [CLS] token

def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    if args.dataset == 'amazon':
        args.graph_type = 'hyperedge'
    elif args.dataset == 'prime':
        args.graph_type = 'triple'

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    if args.save_path is None:
        print("overwritting args.save_path")
        args.save_path = os.path.join(prefix, args.dataset, args.tasks, args.geo)
        if args.geo == 'beta':
            tmp_str = "g-{}-mode-{}-pf-{}-graph-{}".format(args.gamma, args.beta_mode,args.params_frozen, args.graph_added)
        elif args.geo == 'gamma':
            tmp_str = "g-{}-mode-{}-pf-{}-graph-{}".format(args.gamma, args.gamma_mode,args.params_frozen, args.graph_added)

        if args.checkpoint_path is not None:
            args.save_path = args.checkpoint_path
        else:
            args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print ("logging to", args.save_path)
    if not args.do_train: # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    if args.dataset == 'amazon':
        with open('%s/stats.txt'%(args.data_directory+'/'+args.dataset)) as f:
            entrel = f.readlines()
            nprod_entity = int(entrel[0].split(' ')[-1])
            natrr_relation = int(entrel[2].split(' ')[-1])
            natrr_entity = int(entrel[3].split(' ')[-1])
            nentity = nprod_entity + natrr_entity
            nrelation = natrr_relation

            args.nprod = nprod_entity
            args.nattr = natrr_relation
            args.nattrval = natrr_entity
            args.nentity = nentity
            args.nrelation = natrr_relation

        train_queries_unsorted, train_answers, valid_queries_unsorted, valid_easy_answers, valid_hard_answers, \
            test_queries_unsorted, test_easy_answers, test_hard_answers, \
            id2attrval, id2attr, id2prod, id2prod_title = load_data(args, args.tasks)
        
        train_queries_unsorted = ent_merge(args, train_queries_unsorted)
        train_answers = answer_ent_merge(args,train_answers)
        valid_queries_unsorted = ent_merge(args, valid_queries_unsorted)
        valid_hard_answers = answer_ent_merge(args,valid_hard_answers)
        valid_easy_answers = answer_ent_merge(args,valid_easy_answers)  
        test_queries_unsorted = ent_merge(args, test_queries_unsorted)
        test_hard_answers = answer_ent_merge(args,test_hard_answers)
        test_easy_answers = answer_ent_merge(args,test_easy_answers)  
        
        id2ent_title = id_ent_merge(id2attrval, id2prod_title)
        id2rel = id2attr

    elif args.dataset == 'prime':
        with open('%s/stats.txt'%(args.data_directory+'/'+args.dataset)) as f:
            entrel = f.readlines()
            nentity = int(entrel[0].split(' ')[-1])
            nrelation = int(entrel[1].split(' ')[-1])
            nprod_entity = 0

            args.nprod = nprod_entity
            args.nentity = nentity
            args.nrelation = nrelation

        test_queries_unsorted, test_easy_answers, test_hard_answers, \
            id2ent, id2rel, id2ent_title, train_queries_unsorted, train_answers, \
        valid_queries_unsorted, valid_easy_answers, valid_hard_answers = load_data(args, args.tasks)     

    ## id2ent_title, id2rel
    #id2rel, id2ent_title = tokenize_text(id2rel, id2ent_title)

    ## Precompute BERT Embeddings of entities and relations
    f = '/workspace/HybridQA/HybridQE_cross-attention'
    if os.path.exists(os.path.join(f, '{}_ent_bert_embeddings.pt'.format(args.dataset))):
        ent_bert_embeddings = torch.load(os.path.join(f, '{}_ent_bert_embeddings.pt'.format(args.dataset)))
        rel_bert_embeddings = torch.load(os.path.join(f, '{}_rel_bert_embeddings.pt'.format(args.dataset)))
    else:
        # Initialize BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").eval()  # Disable dropout
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)

        # Precompute embeddings for entities and relations
        ent_bert_embeddings = torch.stack([get_bert_embedding(id2ent_title[i], bert_model, tokenizer, device) for i in range(len(id2ent_title))])
        rel_bert_embeddings = torch.stack([get_bert_embedding(id2rel[i], bert_model, tokenizer, device) for i in range(len(id2rel))])

        # Save for later use
        torch.save(ent_bert_embeddings, os.path.join(f, '{}_ent_bert_embeddings.pt'.format(args.dataset)))
        torch.save(rel_bert_embeddings, os.path.join(f, '{}_rel_bert_embeddings.pt'.format(args.dataset)))

    ## Precompute BERT Embeddings of textual queries
    if os.path.exists(os.path.join(f, '{}_id2text_queries.pkl'.format(args.dataset))):
        id2text_queries = pickle.load(open(os.path.join(f,'{}_id2text_queries.pkl'.format(args.dataset)), 'rb'))
        text_query_embeddings = torch.load(os.path.join(f, '{}_textual_query_embeddings.pt'.format(args.dataset)))
        train_queries = pickle.load(open(os.path.join(f,'{}_sorted_train-queries.pkl'.format(args.dataset)), 'rb'))
        valid_queries = pickle.load(open(os.path.join(f,'{}_sorted_valid-queries.pkl'.format(args.dataset)), 'rb'))
        test_queries = pickle.load(open(os.path.join(f,'{}_sorted_test-queries.pkl'.format(args.dataset)), 'rb'))
        #ablation_test_queries = pickle.load(open('/workspace/HybridQA/Datasets/ablation/amazon/Few_Shot_Queries/group_4_queries.pkl', 'rb'))
        #ablation_test_queries = ent_merge(args, ablation_test_queries)

        #text_queries2id = {}
        #for id in id2text_queries:
        #    text_queries2id[id2text_queries[id]] = id

        #test_queries = {}
        #for query_type in ablation_test_queries:
        #    queries = ablation_test_queries[query_type]
        #    new_queries = []
        #    for query in queries:
        #        new_queries.append((query[0], text_queries2id[query[1]]))
        #    test_queries[query_type] = set(new_queries)

        #if args.graph_added:
        #    train_queries[('e',('r','o'))] = set()
        #    queries_1p = copy.deepcopy(train_queries[('e',('r',))])
        #    for query in queries_1p:
        #        if query[1] == 0:
        #            train_queries[('e',('r','o'))].add(query)
        #            train_queries[('e',('r',))].remove(query)
        #else:
        #    queries_1p = copy.deepcopy(train_queries[('e',('r',))])
        #    for query in queries_1p:
        #        if query[1] == 0:
        #            train_queries[('e',('r',))].remove(query)

        if args.do_ablation_on_symbolic:
            ablation_symbolic_queries, ablation_symbolic_query_hard_answers, ablation_symbolic_query_easy_answers = load_ablation_data(args, tasks)
            if args.dataset == 'amazon':
                ablation_symbolic_queries = ent_merge(args, ablation_symbolic_queries)
                ablation_symbolic_query_hard_answers = answer_ent_merge(args, ablation_symbolic_query_hard_answers)
                ablation_symbolic_query_easy_answers = answer_ent_merge(args, ablation_symbolic_query_easy_answers)

            ablation_queries = defaultdict(set)
            ablation_hard_answers = defaultdict()
            ablation_easy_answers = defaultdict()
            for query_type in ablation_symbolic_queries:
                for query in ablation_symbolic_queries[query_type]:
                    ablation_queries[query_structure2query_only[query_type]].add((query,0))
                    ablation_hard_answers[(query, '')] = ablation_symbolic_query_hard_answers[query]
                    ablation_easy_answers[(query, '')] = ablation_symbolic_query_easy_answers[query]
    else:
        # Initialize BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").eval()  # Disable dropout
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)

        # Precompute embeddings for text queries 
        text_questions = set()
        text_queries2id = {}
        id2text_queries = {}
        for query_type in train_queries_unsorted:
            queries = train_queries_unsorted[query_type]
            for query in queries:
                text_questions.add(query[1])
        for query_type in valid_queries_unsorted:
            queries = valid_queries_unsorted[query_type]
            for query in queries:
                text_questions.add(query[1])
        for query_type in test_queries_unsorted:
            queries = test_queries_unsorted[query_type]
            for query in queries:
                text_questions.add(query[1])
        
        for i, query in enumerate(list(text_questions)):
            text_queries2id[query] = i
            id2text_queries[i] = query
        
        train_queries = {}
        for query_type in train_queries_unsorted:
            queries = train_queries_unsorted[query_type]
            new_queries = []
            for i, query in enumerate(list(queries)): 
                new_queries.append((query[0], text_queries2id[query[1]]))
            train_queries[query_type] = set(new_queries)
        
        valid_queries = {}
        for query_type in valid_queries_unsorted:
            queries = valid_queries_unsorted[query_type]
            new_queries = []
            for i, query in enumerate(list(queries)):
                new_queries.append((query[0], text_queries2id[query[1]]))
            valid_queries[query_type] = set(new_queries)
        
        test_queries = {}
        for query_type in test_queries_unsorted:
            queries = test_queries_unsorted[query_type]
            new_queries = []
            for i, query in enumerate(list(queries)):
                new_queries.append((query[0], text_queries2id[query[1]]))
            test_queries[query_type] = set(new_queries)

        textual_query_embeddings = torch.stack([get_bert_embedding(id2text_queries[i], bert_model, tokenizer, device) for i in range(len(id2text_queries))])
        # Save for later use
        torch.save(textual_query_embeddings, os.path.join(f, '{}_textual_query_embeddings.pt'.format(args.dataset)))

        ## dump new train queries 
        pickle.dump(id2text_queries, open(os.path.join(f,'{}_id2text_queries.pkl'.format(args.dataset)), 'wb'))
        pickle.dump(train_queries, open(osp.join(f, '{}_sorted_train-queries.pkl'.format(args.dataset)), 'wb'))
        pickle.dump(valid_queries, open(osp.join(f, '{}_sorted_valid-queries.pkl'.format(args.dataset)), 'wb'))
        pickle.dump(test_queries, open(osp.join(f, '{}_sorted_test-queries.pkl'.format(args.dataset)), 'wb'))

        if args.do_ablation_on_symbolic:
            ablation_symbolic_queries, ablation_symbolic_query_hard_answers, ablation_symbolic_query_easy_answers = load_ablation_data(args, tasks)
            if args.dataset == 'amazon':
                ablation_symbolic_queries = ent_merge(args, ablation_symbolic_queries)
                ablation_symbolic_query_hard_answers = answer_ent_merge(args, ablation_symbolic_query_hard_answers)
                ablation_symbolic_query_easy_answers = answer_ent_merge(args, ablation_symbolic_query_easy_answers)

            ablation_queries = defaultdict(set)
            ablation_hard_answers = defaultdict()
            ablation_easy_answers = defaultdict()
            for query_type in ablation_symbolic_queries:
                for query in ablation_symbolic_queries[query_type]:
                    ablation_queries[query_structure2query_only[query_type]].add((query,0))
                    ablation_hard_answers[(query, '')] = ablation_symbolic_query_hard_answers[query]
                    ablation_easy_answers[(query, '')] = ablation_symbolic_query_easy_answers[query]

    '''
    text_queries2id = {}
    for id in id2text_queries:
        text_queries2id[id2text_queries[id]] = id

    sorted_test_easy_answers = {}
    for query in test_easy_answers:
        sorted_test_easy_answers[(query[0], text_queries2id[query[1]])] = test_easy_answers[query]
    
    sorted_test_hard_answers = {}
    for query in test_hard_answers:
        sorted_test_hard_answers[(query[0], text_queries2id[query[1]])] = test_hard_answers[query]
    
    sorted_train_answers = {}
    for query in train_answers:
        sorted_train_answers[(query[0], text_queries2id[query[1]])] = train_answers[query]

    sorted_valid_easy_answers = {}
    for query in valid_easy_answers:
        sorted_valid_easy_answers[(query[0], text_queries2id[query[1]])] = valid_easy_answers[query]
    
    sorted_valid_hard_answers = {}
    for query in valid_hard_answers:
        sorted_valid_hard_answers[(query[0], text_queries2id[query[1]])] = valid_hard_answers[query]

    pickle.dump(sorted_test_easy_answers, open(osp.join(f, '{}_sorted_test_easy_answers.pkl'.format(args.dataset)), 'wb'))
    pickle.dump(sorted_test_hard_answers, open(osp.join(f, '{}_sorted_test_hard_answers.pkl'.format(args.dataset)), 'wb'))
    pickle.dump(sorted_valid_easy_answers, open(osp.join(f, '{}_sorted_valid_easy_answers.pkl'.format(args.dataset)), 'wb'))
    pickle.dump(sorted_valid_hard_answers, open(osp.join(f, '{}_sorted_valid_hard_answers.pkl'.format(args.dataset)), 'wb'))
    pickle.dump(sorted_train_answers, open(osp.join(f, '{}_sorted_train_answers.pkl'.format(args.dataset)), 'wb'))
    '''

    ## Precompute BERT Embeddings of textual symbolic queries
    if os.path.exists(os.path.join(f, '{}_id2text_symbolic_queries.pkl'.format(args.dataset))):
        id2text_symbolic_queries = pickle.load(open(os.path.join(f,'{}_id2text_symbolic_queries.pkl'.format(args.dataset)), 'rb'))
        text_symbolic_query_embeddings = torch.load(os.path.join(f, '{}_text_symbolic_query_embeddings.pt'.format(args.dataset)))
        text_symbolic_queries2id = {}
        for id in id2text_symbolic_queries:
            text_symbolic_queries2id[id2text_symbolic_queries[id][0]] = id
    else:
        # Initialize BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").eval()  # Disable dropout
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model.to(device)

        # Precompute embeddings for textualized symbolic queries 
        text_symbolic_queries = set()
        text_symbolic_queries2id = {}
        id2text_symbolic_queries = {}

        for query_type in train_queries:
            queries = train_queries[query_type]
            for query in queries:
                text_symbolic_queries.add((query[0], query_name_dict[query_type]))
        for query_type in valid_queries:
            queries = valid_queries[query_type]
            for query in queries:
                text_symbolic_queries.add((query[0], query_name_dict[query_type]))
        for query_type in test_queries_unsorted:
            queries = test_queries[query_type]
            for query in queries:
                text_symbolic_queries.add((query[0], query_name_dict[query_type]))

        for i, query in enumerate(list(text_symbolic_queries)): 
            text_symbolic_queries2id[query] = i
            id2text_symbolic_queries[i] = query
        
        text_symbolic_queries2id = {}
        for id in id2text_symbolic_queries:
            text_symbolic_queries2id[id2text_symbolic_queries[id][0]] = id

        text_symbolic_query_embeddings = torch.stack([get_bert_embedding(encode_query2question(args.dataset, args.nprod, id2text_symbolic_queries[i][0], id2text_symbolic_queries[i][1], id2ent_title, id2rel), bert_model, tokenizer, device) for i in range(len(id2text_symbolic_queries))])
        torch.save(text_symbolic_query_embeddings, os.path.join(f, '{}_text_symbolic_query_embeddings.pt'.format(args.dataset)))
        pickle.dump(id2text_symbolic_queries, open(os.path.join(f,'{}_id2text_symbolic_queries.pkl'.format(args.dataset)), 'wb'))

    ## prepare batch dataset for queries
    if args.do_train:
        train_queries = flatten_query(train_queries)
        train_queries_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainDataset(train_queries, args.nprod, args.nentity, args.nrelation, args.negative_sample_size, train_answers, args.dataset, id2text_queries, text_symbolic_queries2id),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainDataset.collate_fn
                                ))
    
    if args.do_valid:
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
                TestDataset(
                    valid_queries, 
                    args.nprod,
                    args.nentity, 
                    args.nrelation, 
                    args.dataset,
                    text_symbolic_queries2id
                ), 
                batch_size=args.test_batch_size,
                num_workers=args.cpu_num, 
                collate_fn=TestDataset.collate_fn)

    if args.do_test:
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
                TestDataset(
                    test_queries, 
                    args.nprod,
                    args.nentity, 
                    args.nrelation, 
                    args.dataset,
                    text_symbolic_queries2id
                ), 
                batch_size=args.test_batch_size,
                num_workers=args.cpu_num, 
                collate_fn=TestDataset.collate_fn)
        
    if args.do_ablation_on_symbolic:
        ablation_queries = flatten_query(ablation_queries)
        ablation_dataloader = DataLoader(
                TestDataset(
                    ablation_queries, 
                    args.nprod,
                    args.nentity, 
                    args.nrelation, 
                    args.dataset,
                    text_symbolic_queries2id
                ), 
                batch_size=args.test_batch_size,
                num_workers=args.cpu_num, 
                collate_fn=TestDataset.collate_fn)
    
    ## set up model
    model = KGReasoning(
        nprod=args.nprod,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda=args.cuda,
        center_reg=args.center_reg,
        beta_mode = eval_tuple(args.beta_mode),
        gamma_mode=eval_tuple(args.gamma_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict,
        drop=args.drop,
        graph_type=args.graph_type,
        ent_bert_embeddings=ent_bert_embeddings,
        rel_bert_embeddings=rel_bert_embeddings,
        params_frozen=args.params_frozen,
        textual_query_embeddings=text_query_embeddings,
        textual_symbolic_query_embeddings=text_symbolic_query_embeddings
    )
    
    name_to_optimizer = {
        'adam': torch.optim.Adam,
        'adagrad': torch.optim.Adagrad
    }

    assert args.optimizer in name_to_optimizer
    OptimizerClass = name_to_optimizer[args.optimizer]

    if args.cuda:
        model = model.cuda()
        
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = OptimizerClass(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate)
        warm_up_steps = args.max_steps // 2 if args.warm_up_steps is None else args.warm_up_steps

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step 
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []
        # #Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = model.train_step(model, optimizer, train_queries_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            if train_queries_iterator is not None:
                log = model.train_step(model, optimizer, train_queries_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = model.train_step(model, optimizer, train_queries_iterator, args, step)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)
            
            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer, id2text_queries)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer, id2text_queries)
                
                if args.do_ablation_on_symbolic:
                    logging.info('Evaluating on ablation symbolic Dataset...')
                    ablation_all_metrics = evaluate(model, ablation_easy_answers, ablation_hard_answers, args, ablation_dataloader, query_name_dict, 'Ablation', step, writer, id2text_queries)
                    
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)
    
    try:
        print (step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer, id2text_queries)

    if args.do_ablation_on_symbolic:
        logging.info('Evaluating on ablation symbolic Dataset...')
        ablation_all_metrics = evaluate(model, ablation_easy_answers, ablation_hard_answers, args, ablation_dataloader, query_name_dict, 'Ablation', step, writer, id2text_queries)

    logging.info("Training finished!!")


if __name__ == '__main__':
    main(parse_args())