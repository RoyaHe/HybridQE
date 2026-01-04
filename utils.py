import numpy as np
import random
import torch
import time
from collections import defaultdict

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def id_ent_merge(id2attrval, id2prod_title):
    prod_len = len(id2prod_title)
    for id in id2attrval:
        id2prod_title[prod_len+id] = id2attrval[id]
    return id2prod_title

def ent_merge(args, queries):
    nprod_ent = args.nprod
    output_queries = defaultdict(set)
    
    for query_type in queries:
        for query in list(queries[query_type]):
            if type(query[-1]) == str:
                symbolic_query = query[0] 
            else: 
                symbolic_query = query

            if query_type == ('e', ('r',)) or query_type == ('e', ('r','o')):

                symbolic_query = (symbolic_query[0]+nprod_ent, (symbolic_query[1][0],))

            elif query_type == (('e', ('r',)), ('e', ('r',))):
                
                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],)))

            elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',))):

                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],)), 
                                  (symbolic_query[2][0]+nprod_ent, (symbolic_query[2][1][0],)))

            elif query_type == ('e', ('r', 'n')):

                symbolic_query = (symbolic_query[0]+nprod_ent, (symbolic_query[1][0],symbolic_query[1][1]))

            elif query_type == (('e', ('r',)), ('e', ('r', 'n'))):

                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],symbolic_query[1][1][1])))

            elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))):
                
                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],)),
                                  (symbolic_query[2][0]+nprod_ent, (symbolic_query[2][1][0],symbolic_query[2][1][1])))
                
            elif query_type == (('e', ('r',)), ('e', ('r',)), ('u',)):

                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],)),
                                  (symbolic_query[2][0],))

            elif query_type == (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',)):

                symbolic_query = ((symbolic_query[0][0]+nprod_ent, (symbolic_query[0][1][0],)), 
                                  (symbolic_query[1][0]+nprod_ent, (symbolic_query[1][1][0],)),
                                  (symbolic_query[2][0]+nprod_ent, (symbolic_query[2][1][0],)),
                                  (symbolic_query[3][0],))
            
            if type(query[-1]) == str:
                output_queries[query_type].add((symbolic_query, query[1]))
            else: 
                output_queries[query_type].add(symbolic_query)
    return output_queries


def replace_ent(nprod_ent, symbolic_query):
    all_relation_flag = True
    for ele in symbolic_query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(symbolic_query[0]) == int:
            symbolic_query[0] += nprod_ent
        else:
            symbolic_query[0] = replace_ent(nprod_ent, symbolic_query[0])
    else:
        union_flag = False
        if len(symbolic_query[-1]) == 1 and symbolic_query[-1][0] == -1:
            union_flag = True
        for i in range(0, len(symbolic_query)):
            if not union_flag:
                symbolic_query[i] = replace_ent(nprod_ent, symbolic_query[i])
            else:
                if i == len(symbolic_query) - 1:
                    break
                else:
                    symbolic_query[i] = replace_ent(nprod_ent, symbolic_query[i])

    return symbolic_query

def answer_ent_merge(args, answers):
    nprod_ent = args.nprod
    output_answers = defaultdict(set)
    for query in answers:
        if type(query[-1]) == str:
                symbolic_query = query[0] 
        else: 
            symbolic_query = query
        symbolic_query = tuple2list(symbolic_query)  
        symbolic_query = replace_ent(nprod_ent, symbolic_query)
        if type(query[-1]) == str:
            output_answers[(list2tuple(symbolic_query),query[-1])] = answers[query]
        else: 
            output_answers[list2tuple(symbolic_query)] = answers[query]
    return output_answers
        
def encode_query2question(dataset, nprod, query, query_type, id2ent, id2rel):
    nprod = 0
    if dataset == 'amazon':
        output = 'please find out the products with the following attributes:'
        if query_type == '1p':
            output = output + ' and {} is {}.'.format(id2rel[query[1][0]], id2ent[nprod+query[0]])
        if query_type == '1p-only':
            output = output + ' and {} is {}.'.format(id2rel[query[1][0]], id2ent[nprod+query[0]])
        elif query_type == '2i':
            output = output + ' and {} is {}, {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '2i-only':
            output = output + ' and {} is {}, {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '3i':
            output = output + ' and {} is {}, {} is {}, {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
        elif query_type == '3i-only':
            output = output + ' and {} is {}, {} is {}, {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
        elif query_type == '1n':
            output = output + ' and {} is not {}.'.format(id2rel[query[1][0]], id2ent[nprod+query[0]])
        elif query_type == '1n-only':
            output = output + ' and {} is not {}.'.format(id2rel[query[1][0]], id2ent[nprod+query[0]])
        elif query_type == '2in':
            output = output + ' and {} is {}, {} is not {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '2in-only':
            output = output + ' and {} is {}, {} is not {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '3in':
            output = output + ' and {} is {}, {} is {}, {} is not {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
        elif query_type == '3in-only':
            output = output + ' and {} is {}, {} is {}, {} is not {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
        elif query_type == '2u':
            output = output + ' and {} is {}, or {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '3u':
            output = output + ' and {} is {}, or {} is {}, or {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
        elif query_type == '2u-only':
            output = output + ' and {} is {}, or {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]])
        elif query_type == '3u-only':
            output = output + ' and {} is {}, or {} is {}, or {} is {}.'.format(id2rel[query[0][1][0]], id2ent[nprod+query[0][0]], id2rel[query[1][1][0]], id2ent[nprod+query[1][0]], id2rel[query[2][1][0]], id2ent[nprod+query[2][0]])
    elif dataset == 'prime':
        output = 'Please find out items that '
        if query_type == '1p':
            output = output + 'they {} {}.'.format(id2rel[query[1][0]], id2ent[query[0]])
        elif query_type == '2p':
            output = output + 'they {} something that {} {}.'.format(id2rel[query[1][0]], id2rel[query[1][1]], id2ent[query[0]])
        elif query_type == '3p':
            output = output + 'they {} something that {} something that {} {}.'.format(id2rel[query[1][0]], id2rel[query[1][1]], id2rel[query[1][2]], id2ent[query[0]])
        elif query_type == '2i':
            output = output + 'they {} {}, and {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]])
        elif query_type == '3i':
            output = output + 'they {} {}, and {} {}, and {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]], id2rel[query[2][1][0]], id2ent[query[2][0]])
        elif query_type == '1n':
            output = output + 'they do not {} {}.'.format(id2rel[query[1][0]], id2ent[query[0]])
        elif query_type == '2in':
            output = output + 'they {} {}, but do not {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]])
        elif query_type == '3in':
            output = output + 'they {} {} and {} {}, but do not {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]], id2rel[query[2][1][0]], id2ent[query[2][0]])
        elif query_type == '2u':
            output = output + 'they {} {} or {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]])
        elif query_type == '3u':
            output = output + 'they {} {} or {} {}, or {} {}.'.format(id2rel[query[0][1][0]], id2ent[query[0][0]], id2rel[query[1][1][0]], id2ent[query[1][0]], id2rel[query[2][1][0]], id2ent[query[2][0]])
        
    return output