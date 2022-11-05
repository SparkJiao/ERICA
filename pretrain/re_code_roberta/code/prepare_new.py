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

    if not os.path.exists("/data1/private/qinyujia/data_gen/CP"):
        os.mkdir("/data1/private/qinyujia/data_gen/CP")
    json.dump(list_data, open("/data1/private/qinyujia/data_gen/CP/cpdata.json","w"))
    json.dump(rel2scope, open("/data1/private/qinyujia/data_gen/CP/rel2scope.json", 'w'))


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

    if not os.path.exists("/data1/private/qinyujia/data_gen/MTB"):
        os.mkdir("/data1/private/qinyujia/data_gen/MTB")
    json.dump(entpair2negpair, open("/data1/private/qinyujia/data_gen/MTB/entpair2negpair.json","w"))
    json.dump(entpair2scope, open("/data1/private/qinyujia/data_gen/MTB/entpair2scope.json", "w"))
    json.dump(list_data, open("/data1/private/qinyujia/data_gen/MTB/mtbdata.json", "w"))

def process_data_for_CP_R(data):
    wikidata = {}
    example = {}
    r2n = json.load(open('../data/relation_name_desc.json', 'r'))
    r2n = {k: v['name'] for k,v in r2n.items()}
    for f_name in ['train.txt', 'test.txt', 'dev.txt']:
        with open(os.path.join('../data/wiki80', f_name)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                wikidata[' '.join(ins['token'])] = 0

    all_entities = {}
    entity2relation = {}
    count = 0
    all_relation = []
    washed_data = {}
    ent2sen = {}

    c = 0
    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            if key not in example:
                if key in r2n:
                    example[key] = [sen]
            elif len(example[key]) < 5:
                example[key].append(sen)

            head = sen['h']['id']
            tail = sen['t']['id']
            if ' '.join(sen['tokens']) in wikidata:
                c += 1
                continue
            if filter_sentence(sen):
                continue
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
    example = {r2n[k]:v for k,v in example.items()}

    print(c)
    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)

    print(len(all_relation))
    relation2relation_q = {}
    relation2sen_q = {}
    relation2sen_soft_q = {}
    relation2relation_t = {}
    relation2sen_t = {}
    relation2sen_soft_t = {}

    example_2 = {}
    for e1 in all_entities:
        for e2 in all_entities[e1]:
            if e2 not in all_entities:
                continue

            for e3 in all_entities[e2]:
                r1 = entity2relation[e1+'#'+e2]
                r2 = entity2relation[e2+'#'+e3]
                if r1+'#'+r2 not in relation2sen_soft_t:
                    relation2sen_soft_t[r1+'#'+r2] = [['#'.join([e1, e2]), '#'.join([e2, e3])]]
                else:
                    relation2sen_soft_t[r1+'#'+r2].append(['#'.join([e1, e2]), '#'.join([e2, e3])])
                if e3 in all_entities[e1]:
                    r3 = entity2relation[e1+'#'+e3]
                    if r1 in r2n and r2 in r2n and r3 in r2n:
                        if '#'.join([r2n[r1], r2n[r2], r2n[r3]]) not in example_2:
                            example_2['#'.join([r2n[r1], r2n[r2], r2n[r3]])] = [[e1, e2, e3]]
                        elif len(example_2['#'.join([r2n[r1], r2n[r2], r2n[r3]])]) < 5:
                            example_2['#'.join([r2n[r1], r2n[r2], r2n[r3]])].append([e1, e2, e3])

                    if r1+'#'+r2+'#'+r3 not in relation2relation_t:
                        relation2relation_t[r1+'#'+r2+'#'+r3] = 1
                        relation2sen_t[r1+'#'+r2+'#'+r3] = [['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e1, e3])]]
                    else:
                        relation2relation_t[r1+'#'+r2+'#'+r3] += 1
                        relation2sen_t[r1+'#'+r2+'#'+r3].append(['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e1, e3])])

                if e3 not in all_entities:
                    continue
                for e4 in all_entities[e3]:
                    r1 = entity2relation[e1+'#'+e2]
                    r2 = entity2relation[e2+'#'+e3]
                    r3 = entity2relation[e3+'#'+e4]
                    if r1+'#'+r2+'#'+r3 not in relation2sen_soft_q:
                        relation2sen_soft_q[r1+'#'+r2+'#'+r3] = [['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e3, e4])]]
                    else:
                        relation2sen_soft_q[r1+'#'+r2+'#'+r3].append(['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e3, e4])])
                    if e4 in all_entities[e1]:
                        r4 = entity2relation[e1+'#'+e4]
                        if r1+'#'+r2+'#'+r3+'#'+r4 not in relation2relation_q:
                            relation2relation_q[r1+'#'+r2+'#'+r3+'#'+r4] = 1
                            relation2sen_q[r1+'#'+r2+'#'+r3+'#'+r4] = [['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e3, e4]), '#'.join([e1, e4])]]
                        else:
                            relation2relation_q[r1+'#'+r2+'#'+r3+'#'+r4] += 1
                            relation2sen_q[r1+'#'+r2+'#'+r3+'#'+r4].append(['#'.join([e1, e2]), '#'.join([e2, e3]), '#'.join([e3, e4]), '#'.join([e1, e4])])


    print('before filtering, t: %d - %d, q: %d - %d'%(len(relation2relation_t), sum([v for v in relation2relation_t.values()]), len(relation2relation_q), sum([v for v in relation2relation_q.values()])))
    threshold_t = 50
    threshold_q = 500

    relation2relation_t = {k: v for k,v in relation2relation_t.items() if v > threshold_t and all([kk in washed_data for kk in k.split('#')])}
    relation2relation_q = {k: v for k,v in relation2relation_q.items() if v > threshold_q and all([kk in washed_data for kk in k.split('#')])}
    print('after filtering, t: %d - %d, q: %d - %d'%(len(relation2relation_t), sum([v for v in relation2relation_t.values()]), len(relation2relation_q), sum([v for v in relation2relation_q.values()])))

    relation2sen_t = {k: v for k,v in relation2sen_t.items() if k in relation2relation_t}
    relation2sen_q = {k: v for k,v in relation2sen_q.items() if k in relation2relation_q}

    r_map_t = {'#'.join(k.split('#')[:2]): k.split('#')[2] for k in relation2relation_t}
    r_map_q = {'#'.join(k.split('#')[:3]): k.split('#')[3] for k in relation2relation_q}


    relation2sen_soft_t = {k: v for k,v in relation2sen_soft_t.items() if k in r_map_t}
    relation2sen_soft_q = {k: v for k,v in relation2sen_soft_q.items() if k in r_map_q}

    ent_tag = {}
    for v in list(relation2sen_soft_t.values()) + list(relation2sen_soft_q.values()):
        for vv in v:
            for vvv in vv:
                if vvv not in ent_tag:
                    ent_tag[vvv] = 0

    ent2sen = {k:v for k,v in ent2sen.items() if k in ent_tag}


    if not os.path.exists("../data/CP_R_large"):
        os.mkdir("../data/CP_R_large")
    json.dump(list_data, open("../data/CP_R_large/cpdata.json","w"))
    json.dump(rel2scope, open("../data/CP_R_large/rel2scope.json", 'w'))
    json.dump(ent2sen, open("../data/CP_R_large/ent2sen.json", 'w'))
    json.dump(r_map_t, open("../data/CP_R_large/r_map_t.json", 'w'))
    json.dump(r_map_q, open("../data/CP_R_large/r_map_q.json", 'w'))
    json.dump(relation2sen_soft_t, open("../data/CP_R_large/relation2sen_soft_t.json", 'w'))
    json.dump(relation2sen_soft_q, open("../data/CP_R_large/relation2sen_soft_q.json", 'w'))

    #json.dump(list_data, open("../data/CP_R/cpdata_debug.json","w"))
    #json.dump(rel2scope, open("../data/CP_R/rel2scope_debug.json", 'w'))
    #json.dump(ent2sen, open("../data/CP_R/ent2sen_debug.json", 'w'))
    #json.dump(r_map_t, open("../data/CP_R/r_map_t_debug.json", 'w'))
    #json.dump(r_map_q, open("../data/CP_R/r_map_q_debug.json", 'w'))
    #json.dump(relation2sen_soft_t, open("../data/CP_R/relation2sen_soft_t_debug.json", 'w'))
    #json.dump(relation2sen_soft_q, open("../data/CP_R/relation2sen_soft_q_debug.json", 'w'))

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
        data = json.load(open("/data1/private/qinyujia/data_gen/exclude_fewrel_distant.json"))
        process_data_for_CP(data)

    elif args.dataset == "mtb":
        data = json.load(open("/data1/private/qinyujia/data_gen/exclude_fewrel_distant.json"))
        process_data_for_MTB(data)

    elif args.dataset == "cp_r":
        data = json.load(open("../data/distant_all.json"))
        process_data_for_CP_R(data)

