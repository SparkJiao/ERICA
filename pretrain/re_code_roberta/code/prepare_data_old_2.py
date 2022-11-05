import json 
import random
import os 
import sys 
import pdb 
import re 
import torch
import argparse
import numpy as np 
from tqdm import trange
from collections import Counter, defaultdict


def filter_sentence(sentence):
    """Filter sentence.
    
    Filter sentence:
        - head mention equals tail mention
        - head mentioin and tail mention overlap

    Args:
        sentence: A python dict.
            sentence example:
            {
                'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                'r': 'P1'
            }

    Returns:
        True or False. If the sentence contains abnormal conditions 
        above, return True. Else return False

    Raises:
        If sentence's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    head_pos = sentence["h"]["pos"][0]
    tail_pos = sentence["t"]["pos"][0]
    
    if sentence["h"]["name"] == sentence["t"]["name"]:  # head mention equals tail mention
        return True

    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]: # head mentioin and tail mention overlap
        return True
    
    if tail_pos[0] >= head_pos[0] and tail_pos[0] <= head_pos[-1]: # head mentioin and tail mention overlap
        return True  

    return False



def process_data_for_CP(data):
    """Process data for CP. 

    This function will filter NA relation, abnormal sentences,
    and relation of which sentence number is less than 2(This relation
    can't form positive sentence pair).

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        But this function will save two json-formatted files:
            - list_data: A list of sentences.
            - rel2scope: A python dict whose key is relation and value is 
                a scope which is left-closed-right-open `[)`. All sentences 
                in a same scope share the same relation.
            
            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                
                - rel2scope:
                    {
                        'P10': [0, 233],
                        'P1212': [233, 1000],
                        ....
                    }
        
    Raises:
        If data's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    washed_data = {}
    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            if filter_sentence(sen):
                continue
            rel_sentence_list.append(sen)
        if len(rel_sentence_list) < 2:
            continue        
        washed_data[key] = rel_sentence_list

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    
    if not os.path.exists("../data/CP"):
        os.mkdir("../data/CP")
    json.dump(list_data, open("../data/CP/cpdata.json","w"))
    json.dump(rel2scope, open("../data/CP/rel2scope.json", 'w'))


def process_data_for_MTB(data):
    """Process data for MTB. 

    This function will filter abnormal sentences, and entity pair of which 
    sentence number is less than 2(This entity pair can't form positive sentence pair).

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'token': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        But this function will save three json-formatted files:
            - list_data: A list of sentences.
            - entpair2scope: A python dict whose key is `head_id#tail_id` and value is 
                a scope which is left-closed-right-open `[)`. All sentences in one same 
                scope share the same entity pair
            - entpair2negpair: A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). 

            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                - entpair2scope:
                    {
                        'Q1234#Q2356': [0, 233],
                        'Q135656#Q10': [233, 1000],
                        ....
                    }
                - entpair2negpair:
                    {
                        'Q1234#Q2356': ['Q1234#Q3560','Q923#Q2356', 'Q1234#Q100'],
                        'Q135656#Q10': ['Q135656#Q9', 'Q135656#Q10010', 'Q2666#Q10']
                    }
        
    Raises:
        If data's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    # Maximum number of sentences sharing the same entity pair.
    # This parameter is set for limit the bias towards popular 
    # entity pairs which have many sentences. Of cource, you can
    # change this parameter, but in our expriment, we use 8.
    max_num = 8 

    # We change the original data's format. The ent_data is 
    # a python dict of which key is `head_id#tail_id` and value
    # is sentences which hold this same entity pair.
    ent_data = defaultdict(list)
    for key in data.keys():
        for sentence in data[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            ent_data[head + "#" + tail].append(sentence)

    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        if len(ent_data[key]) < 2:
            continue
        list_data.extend(ent_data[key][0:max_num])
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)

    # We will pre-generate `hard` nagative samples. The entpair2negpair
    # is a python dict of which key is `head_id#tail_id`. And the value of the dict
    # is the same format as key, but head_id or tail_id is different(only one id is 
    # different). 
    entpair2negpair = defaultdict(list)
    entpairs = list(entpair2scope.keys())
    entpairs.sort(key=lambda a: a.split("#")[0])
    for i in range(len(entpairs)):
        head = entpairs[i].split("#")[0]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[0] != head:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    entpairs.sort(key=lambda a: a.split("#")[1])
    for i in range(len(entpairs)):
        tail = entpairs[i].split("#")[1]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[1] != tail:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    if not os.path.exists("../data/MTB"):
        os.mkdir("../data/MTB")
    json.dump(entpair2negpair, open("../data/MTB/entpair2negpair.json","w"))
    json.dump(entpair2scope, open("../data/MTB/entpair2scope.json", "w"))
    json.dump(list_data, open("../data/MTB/mtbdata.json", "w"))

def process_data_for_CP_new(data):
    '''
    #only for debugging, will delete later
    data_n = {}
    count = 0
    for k in data:
        if count >= 200:
            break
        count += 1
        data_n[k] = data[k]
    data = data_n
    #only for debugging, will delete later
    '''

    all_entities = {}
    entity2relation = {}
    count = 0
    all_relation = []
    washed_data = {}
    ent2sen = {}

    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            head = sen['h']['id']
            tail = sen['t']['id']
            if filter_sentence(sen):
                continue
            assert sen['r'] == key
            rel_sentence_list.append(sen)
            if head+'#'+tail not in ent2sen:
                ent2sen[head+'#'+tail] = [sen]
            else:
                ent2sen[head+'#'+tail].append(sen)
            if head +'#'+tail not in entity2relation:
                entity2relation[head+'#'+tail] = sen['r']
            if head not in all_entities:
                all_entities[head] = [tail]
            elif tail not in all_entities[head]:
                all_entities[head].append(tail)
        if sen['r'] not in all_relation:
            all_relation.append(sen['r'])
        if len(rel_sentence_list) < 2:
            continue
        washed_data[key] = rel_sentence_list

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)

    print(len(all_relation))
    relation2relation = {}
    relation2sen = {}
    relation2sen_soft = {}
    for e1 in all_entities:
        for e2 in all_entities[e1]:
            if e2 not in all_entities:
                continue
            for e3 in all_entities[e2]:
                r1 = entity2relation[e1+'#'+e2]
                r2 = entity2relation[e2+'#'+e3]
                if r1+'#'+r2 not in relation2sen_soft:
                    relation2sen_soft[r1+'#'+r2] = [[ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3]]]
                else:
                    relation2sen_soft[r1+'#'+r2].append([ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3]])
                if e3 in all_entities[e1]:
                    r3 = entity2relation[e1+'#'+e3]
                    if r1+'#'+r2+'#'+r3 not in relation2relation:
                        relation2relation[r1+'#'+r2+'#'+r3] = 1
                        relation2sen[r1+'#'+r2+'#'+r3] = [[ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e1+'#'+e3]]]
                    else:
                        relation2relation[r1+'#'+r2+'#'+r3] += 1
                        relation2sen[r1+'#'+r2+'#'+r3].append([ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e1+'#'+e3]])
    
    print('before filtering')
    print(len(relation2relation))
    print(sum([v for v in relation2relation.values()]))
    threshold = 20
    relation2relation = {k: v for k,v in relation2relation.items() if v > threshold and all([kk in washed_data for kk in k.split('#')])}
    print('after filtering')
    print(len(relation2relation))
    print(sum([v for v in relation2relation.values()]))
    relation2sen = {k: v for k,v in relation2sen.items() if k in relation2relation}
    r_simple = ['#'.join(k.split('#')[:2]) for k in relation2relation.keys()]
    r_map = {'#'.join(k.split('#')[:2]): k.split('#')[2] for k in relation2relation.keys()}

    relation2sen_soft = {k: v for k,v in relation2sen_soft.items() if k in r_simple}
    n = 0
    for v in relation2sen.values():
        for vv in v:
            n += len(vv[0])*len(vv[1])*len(vv[2])
    print(n)
    n = 0
    for v in relation2sen_soft.values():
        for vv in v:
            n += len(vv[0])*len(vv[1])
    print(n)
    print(sum([len(v) for v in relation2sen_soft.values()]))
    
    print("------------")
    n1 = 0
    n2 = 0
    n3 = 0
    relation_count = {}
    for e1 in all_entities:
        for e2 in all_entities[e1]:
            if e2 not in all_entities:
                continue
            for e3 in all_entities[e2]:
                r1 = entity2relation[e1+'#'+e2]
                r2 = entity2relation[e2+'#'+e3]
                if '#'.join([r1, r2]) in r_simple:
                    if '#'.join([r1, r2]) not in relation_count:
                        relation_count['#'.join([r1, r2])] = [1, 0]
                    else:
                        relation_count['#'.join([r1, r2])][0] += 1
                    n1 += 1
                    if e3 in all_entities[e1]:
                        relation_count['#'.join([r1, r2])][1] += 1
                        n2 += 1
                        if '#'.join([r1, r2]) in r_map:
                            if entity2relation['#'.join([e1, e3])] == r_map['#'.join([r1, r2])]:
                                n3 += 1
    nn = 0
    for k in relation_count:
        print(k)
        print(float(relation_count[k][1]) / float(relation_count[k][0]))
        if float(relation_count[k][1]) / float(relation_count[k][0]) > 0.6:
            nn += 1
    print(nn)
    exit()
    print(n1)
    print(n2)
    print(n3)
    
    exit()
    if not os.path.exists("../data/CP_R"):
        os.mkdir("../data/CP_R")
    json.dump(list_data, open("../data/CP_R/cpdata.json","w"))
    json.dump(rel2scope, open("../data/CP_R/rel2scope.json", 'w'))
    json.dump(relation2sen, open("../data/CP_R/relation2sen.json", 'w'))
    json.dump(relation2sen_soft, open("../data/CP_R/relation2sen_soft.json", 'w'))

    #json.dump(list_data, open("../data/CP_R/cpdata_debug.json","w"))
    #json.dump(rel2scope, open("../data/CP_R/rel2scope_debug.json", 'w'))
    #json.dump(relation2sen, open("../data/CP_R/relation2sen_debug.json", 'w'))
    #json.dump(relation2sen_soft, open("../data/CP_R/relation2sen_soft_debug.json", 'w'))

def process_data_for_CP_new_quadra(data):
    all_entities = {}
    entity2relation = {}
    count = 0
    all_relation = []
    washed_data = {}
    ent2sen = {}

    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            head = sen['h']['id']
            tail = sen['t']['id']
            if filter_sentence(sen):
                continue
            assert sen['r'] == key
            rel_sentence_list.append(sen)
            if head+'#'+tail not in ent2sen:
                ent2sen[head+'#'+tail] = [sen]
            else:
                ent2sen[head+'#'+tail].append(sen)
            if head +'#'+tail not in entity2relation:
                entity2relation[head+'#'+tail] = sen['r']
            if head not in all_entities:
                all_entities[head] = [tail]
            elif tail not in all_entities[head]:
                all_entities[head].append(tail)
        if sen['r'] not in all_relation:
            all_relation.append(sen['r'])
        if len(rel_sentence_list) < 2:
            continue
        washed_data[key] = rel_sentence_list

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)

    print(len(all_relation))
    relation2relation = {}
    relation2sen = {}
    relation2sen_soft = {}
    for e1 in all_entities:
        for e2 in all_entities[e1]:
            if e2 not in all_entities:
                continue
            for e3 in all_entities[e2]:
                if e3 not in all_entities:
                    continue
                for e4 in all_entities[e3]:
                    r1 = entity2relation[e1+'#'+e2]
                    r2 = entity2relation[e2+'#'+e3]
                    r3 = entity2relation[e3+'#'+e4]
                    if r1+'#'+r2+'#'+r3 not in relation2sen_soft:
                        relation2sen_soft[r1+'#'+r2+'#'+r3] = [[ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e3+'#'+e4]]]
                    else:
                        relation2sen_soft[r1+'#'+r2+'#'+r3].append([ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e3+'#'+e4]])
                    if e4 in all_entities[e1]:
                        r4 = entity2relation[e1+'#'+e4]
                        if r1+'#'+r2+'#'+r3+'#'+r4 not in relation2relation:
                            relation2relation[r1+'#'+r2+'#'+r3+'#'+r4] = 1
                            relation2sen[r1+'#'+r2+'#'+r3+'#'+r4] = [[ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e3+'#'+e4], ent2sen[e1+'#'+e4]]]
                        else:
                            relation2relation[r1+'#'+r2+'#'+r3+'#'+r4] += 1
                            relation2sen[r1+'#'+r2+'#'+r3+'#'+r4].append([ent2sen[e1+'#'+e2], ent2sen[e2+'#'+e3], ent2sen[e3+'#'+e4], ent2sen[e1+'#'+e4]])
    
    print('before filtering')
    print(len(relation2relation))
    print(sum([v for v in relation2relation.values()]))
    threshold = 20
    relation2relation = {k: v for k,v in relation2relation.items() if v > threshold and all([kk in washed_data for kk in k.split('#')])}
    print('after filtering')
    print(len(relation2relation))
    print(sum([v for v in relation2relation.values()]))
    relation2sen = {k: v for k,v in relation2sen.items() if k in relation2relation}
    r_simple = ['#'.join(k.split('#')[:3]) for k in relation2relation.keys()]

    relation2sen_soft = {k: v for k,v in relation2sen_soft.items() if k in r_simple}
    
    if not os.path.exists("../data/CP_R"):
        os.mkdir("../data/CP_R")
    json.dump(list_data, open("../data/CP_R/cpdata_quadra.json","w"))
    json.dump(rel2scope, open("../data/CP_R/rel2scope_quadra.json", 'w'))
    json.dump(relation2sen, open("../data/CP_R/relation2sen_quadra.json", 'w'))
    json.dump(relation2sen_soft, open("../data/CP_R/relation2sen_soft_quadra.json", 'w'))

    #json.dump(list_data, open("../data/CP_R/cpdata_debug.json","w"))
    #json.dump(rel2scope, open("../data/CP_R/rel2scope_debug.json", 'w'))
    #json.dump(relation2sen, open("../data/CP_R/relation2sen_debug.json", 'w'))
    #json.dump(relation2sen_soft, open("../data/CP_R/relation2sen_soft_debug.json", 'w'))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="{mtb,cp}")
    args = parser.parse_args()
    set_seed(42)

    if args.dataset == "cp":
        data = json.load(open("../data/exclude_fewrel_distant.json"))
        process_data_for_CP(data)

    elif args.dataset == "mtb":
        data = json.load(open("../data/exclude_fewrel_distant.json"))
        process_data_for_MTB(data)

    elif args.dataset == "cp_3":
        data = json.load(open("../data/exclude_fewrel_distant.json"))
        process_data_for_CP_new(data)

    elif args.dataset == "cp_4":
        data = json.load(open("../data/exclude_fewrel_distant.json"))
        process_data_for_CP_new_quadra(data)
    
    























