import json 
import random
import os 
import sys 
sys.path.append("..")
import pdb 
import re 
import pdb 
import math 
import torch
import numpy as np  
from collections import Counter
from torch.utils import data
from utils import EntityMarker
import re


class CPDataset(data.Dataset):
    """Overwritten class Dataset for model CP.

    This class prepare data for training of CP.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for CP.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path 
        self.args = args 
        if args.debug == 0:
            data = json.load(open(os.path.join(path, "cpdata.json")))
            rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
            self.rel2scope = rel2scope
        elif args.debug == 1:
            data = json.load(open(os.path.join(path, "cpdata_debug.json")))
            rel2scope = json.load(open(os.path.join(path, "rel2scope_debug.json")))
            self.rel2scope = rel2scope
        entityMarker = EntityMarker()

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        self.h_pos_l = np.zeros((len(data)), dtype=int)
        self.t_pos_l = np.zeros((len(data)), dtype=int)

        # Distant supervised label for sentence.
        # Sentences whose label are the same in a batch 
        # is positive pair, otherwise negative pair.
        for i, rel in enumerate(rel2scope.keys()):
            scope = rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i

        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt, ph_l, pt_l = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag, 0)
            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph) 
            self.t_pos[i] = min(args.max_length-1, pt)
            self.h_pos_l[i] = min(args.max_length, ph_l)
            self.t_pos_l[i] = min(args.max_length, pt_l)
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
        # Samples positive pair dynamically. 
        self.__sample__()
    
    def __pos_pair__(self, scope):
        """Generate positive pair.

        Args:
            scope: A scope in which all sentences' label are the same.
                scope example: [0, 12]

        Returns:
            all_pos_pair: All positive pairs. 
            ! IMPORTTANT !
            Given that any sentence pair in scope is positive pair, there
            will be totoally (N-1)N/2 pairs, where N equals scope[1] - scope[0].
            The positive pair's number is proportional to N^2, which will cause 
            instance imbalance. And If we consider all pair, there will be a huge 
            number of positive pairs.
            So we sample positive pair which is proportional to N. And in different epoch,
            we resample sentence pair, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))
        
        # shuffle bag to get different pairs
        random.shuffle(pos_scope)   
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(index)
            if (i+1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair
    
    def __sample__(self):
        """Samples positive pairs.

        After sampling, `self.pos_pair` is all pairs sampled.
        `self.pos_pair` example: 
                [
                    [0,2],
                    [1,6],
                    [12,25],
                    ...
                ]
        """
        #rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        rel2scope = self.rel2scope
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        print("Postive pair's number is %d" % len(self.pos_pair))

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Get training instance.

        Overwitten function.
        
        Args:
            index: Instance index.
        
        Return:
            input: Tokenized word id.
            mask: Attention mask for bert. 0 means masking, 1 means not masking.
            label: Label for sentence.
            h_pos: Position of head entity.
            t_pos: Position of tail entity.
        """
        bag = self.pos_pair[index]
        input = np.zeros((self.args.max_length * 2), dtype=int)
        mask = np.zeros((self.args.max_length * 2), dtype=int)
        label = np.zeros((2), dtype=int)
        h_pos = np.zeros((2), dtype=int)
        t_pos = np.zeros((2), dtype=int)
        h_pos_l = np.zeros((2), dtype=int)
        t_pos_l = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            input[i*self.args.max_length : (i+1)*self.args.max_length] = self.tokens[ind]
            mask[i*self.args.max_length : (i+1)*self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]
            h_pos[i] = self.h_pos[ind]
            t_pos[i] = self.t_pos[ind]
            h_pos_l[i] = self.h_pos_l[ind]
            t_pos_l[i] = self.t_pos_l[ind]

        return input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l

class CP_R_Dataset(data.Dataset):
    """Overwritten class Dataset for model CP.

    This class prepare data for training of CP.
    """
    def __init__(self, path, args):
        self.path = path
        self.args = args

        if args.debug == 0:
            self.rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
            data = json.load(open(os.path.join(path, "cpdata.json")))
            self.relation2sen_soft_t = json.load(open(os.path.join(path, "relation2sen_soft_t.json")))
            self.relation2sen_soft_q = json.load(open(os.path.join(path, "relation2sen_soft_q.json")))
            r_map_t = json.load(open(os.path.join(path, "r_map_t.json")))
            r_map_q = json.load(open(os.path.join(path, "r_map_q.json")))
            self.ent2sen = json.load(open(os.path.join(path, "ent2sen.json")))
        elif args.debug == 1:
            self.rel2scope = json.load(open(os.path.join(self.path, "rel2scope_debug.json")))
            data = json.load(open(os.path.join(path, "cpdata_debug.json")))
            self.relation2sen_soft_t = json.load(open(os.path.join(path, "relation2sen_soft_t_debug.json")))
            self.relation2sen_soft_q = json.load(open(os.path.join(path, "relation2sen_soft_q_debug.json")))
            r_map_t = json.load(open(os.path.join(path, "r_map_t_debug.json")))
            r_map_q = json.load(open(os.path.join(path, "r_map_q_debug.json")))
            self.ent2sen = json.load(open(os.path.join(path, "ent2sen_debug.json")))

        self.relation_map_t = r_map_t
        self.relation_map_q = r_map_q

        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")

        self.entityMarker = EntityMarker()
        self.idx2token = {v: k for k,v in self.entityMarker.tokenizer.vocab.items()}

        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        self.h_pos_l = np.zeros((len(data)), dtype=int)
        self.t_pos_l = np.zeros((len(data)), dtype=int)
        
        self.rel2id = {}
        for i, rel in enumerate(self.rel2scope.keys()):
            self.rel2id[rel] = i
            scope = self.rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i
        
        #single sentence
        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt, ph_l, pt_l = self.entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag, True)

            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph)
            self.t_pos[i] = min(args.max_length-1, pt)
            self.h_pos_l[i] = min(args.max_length, ph_l)
            self.t_pos_l[i] = min(args.max_length, pt_l)
        
        self.ent2sen_tok = {}

        for k in self.ent2sen:
            self.ent2sen_tok[k] = []
            for sentence in self.ent2sen[k]:
                h_p = sentence["h"]["pos"][0]
                t_p = sentence["t"]["pos"][0]
                f_text, p_head, p_tail = self.entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, False, False, False)
                self.ent2sen_tok[k].append([f_text, p_head, p_tail, sentence['r']])
        
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % self.entityMarker.err)
        # Samples positive pair dynamically. 

        def get_pos_test_t(relation2sen_soft_t, ent2sen_tok):            
            all_pos_pair = []
            for i, item_1 in enumerate(relation2sen_soft_t):
                h_flag = random.random() > self.args.alpha
                t_flag = random.random() > self.args.alpha
                hop_flag = random.random() > self.args.alpha
                e1 = self.get_sen(random.choice(ent2sen_tok[item_1[0]]), h_flag, hop_flag, 1)
                e2 = self.get_sen(random.choice(ent2sen_tok[item_1[1]]), hop_flag, t_flag, 3)
                all_pos_pair.append([e1, e2])
            return all_pos_pair

        #get fake test data
        self.pos_R_test_t = []
        for rel in self.relation_map_t:
            if len(self.relation2sen_soft_t[rel]) < 500:
                continue
            relation2sen_soft_t_test = []
            for _ in range(50):
                relation2sen_soft_t_test.append(self.relation2sen_soft_t[rel].pop())
            pos_R = get_pos_test_t(relation2sen_soft_t_test, self.ent2sen_tok)
            self.pos_R_test_t.extend(pos_R)
        self.neg_test = []
        self.ignore_id = []
        for rel in self.rel2scope:
            if self.rel2scope[rel][1] - self.rel2scope[rel][0] < 100:
                continue
            for _ in range(30):
                idx = random.randint(self.rel2scope[rel][0], self.rel2scope[rel][1]-1)
                while(idx in self.ignore_id):
                    idx = random.randint(self.rel2scope[rel][0], self.rel2scope[rel][1]-1)
                self.neg_test.append(idx)
                self.ignore_id.append(idx)

        self.__sample__()
    
    def get_test_set(self):
        pos_R_test_t, neg_test = self.pos_R_test_t, self.neg_test
        #prepare neg_test
        input = np.zeros((len(neg_test), self.args.max_length), dtype=int)
        mask = np.zeros((len(neg_test), self.args.max_length), dtype=int)
        label = np.zeros((len(neg_test)), dtype=int)
        h_pos = np.zeros((len(neg_test)), dtype=int)
        t_pos = np.zeros((len(neg_test)), dtype=int)
        h_pos_l = np.zeros((len(neg_test)), dtype=int)
        t_pos_l = np.zeros((len(neg_test)), dtype=int)

        for i, idx in enumerate(neg_test):
            input[i, :self.args.max_length] = self.tokens[idx]
            mask[i, :self.args.max_length] = self.mask[idx]
            label[i] = self.label[idx]
            h_pos[i] = self.h_pos[idx]
            t_pos[i] = self.t_pos[idx]
            h_pos_l[i] = self.h_pos_l[idx]
            t_pos_l[i] = self.t_pos_l[idx]
        
        input = torch.from_numpy(input)
        mask = torch.from_numpy(mask)
        label = torch.from_numpy(label)
        h_pos = torch.from_numpy(h_pos)
        t_pos = torch.from_numpy(t_pos)
        h_pos_l = torch.from_numpy(h_pos_l)
        t_pos_l = torch.from_numpy(t_pos_l)

        #prepare pos_test_t
        input_R = np.zeros((len(pos_R_test_t), self.args.max_length * 2), dtype=int)
        mask_R = np.zeros((len(pos_R_test_t), self.args.max_length * 2), dtype=int)
        label_R = np.zeros((len(pos_R_test_t)), dtype=int)
        h_pos_R = np.zeros((len(pos_R_test_t)), dtype=int)
        t_pos_R = np.zeros((len(pos_R_test_t)), dtype=int)
        h_pos_l_R = np.zeros((len(pos_R_test_t)), dtype=int)
        t_pos_l_R = np.zeros((len(pos_R_test_t)), dtype=int)

        for i, ind in enumerate(pos_R_test_t):
            ids_1, ph_1, pt_1, ph_l_1, pt_l_1, r_1 = ind[0]
            ids_2, ph_2, pt_2, ph_l_2, pt_l_2, r_2 = ind[1]
            ids = ids_1 + ids_2
            ph = min(self.args.max_length * 2 - 1, ph_1)
            pt = min(self.args.max_length * 2 - 1, len(ids_1) + pt_2)
            ph_l = min(self.args.max_length * 2, ph_l_1)
            pt_l = min(self.args.max_length * 2, len(ids_1) + pt_l_2)
            length = min(len(ids), self.args.max_length * 2)

            input_R[i, :length] = ids[: length]
            mask_R[i, :length] = 1

            label_R[i] = self.rel2id[self.relation_map_t['#'.join([r_1, r_2])]]
            h_pos_R[i] = ph
            t_pos_R[i] = pt
            h_pos_l_R[i] = ph_l
            t_pos_l_R[i] = pt_l
            
            assert ph < ph_l
            assert pt < pt_l

        input_R = torch.from_numpy(input_R)
        mask_R = torch.from_numpy(mask_R)
        label_R = torch.from_numpy(label_R)
        h_pos_R = torch.from_numpy(h_pos_R)
        t_pos_R = torch.from_numpy(t_pos_R)
        h_pos_l_R = torch.from_numpy(h_pos_l_R)
        t_pos_l_R = torch.from_numpy(t_pos_l_R)
        
        single_sentence = (input.to(self.args.device), mask.to(self.args.device), label, h_pos.to(self.args.device), t_pos.to(self.args.device), h_pos_l.to(self.args.device), t_pos_l.to(self.args.device))
        two_hop_pos = (input_R.to(self.args.device), mask_R.to(self.args.device), label_R, h_pos_R.to(self.args.device), t_pos_R.to(self.args.device), h_pos_l_R.to(self.args.device), t_pos_l_R.to(self.args.device))
        return single_sentence, two_hop_pos

    def __pos_pair__(self, scope):
        pos_scope = list(range(scope[0], scope[1]))
        
        random.shuffle(pos_scope)   
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(('1', index))
            if (i+1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair

    def get_sen(self, e, h_flag, t_flag, sen_type = 1):
        f_text, p_head, p_tail, r = e

        if h_flag:
            if sen_type == 1:
                f_text = self.h_pattern.sub("[unused0] [unused4] [unused1]", f_text)
            elif sen_type == 2:
                f_text = self.h_pattern.sub("[unused0] [unused6] [unused1]", f_text)
            elif sen_type == 3:
                f_text = self.h_pattern.sub("[unused0] [unused6] [unused1]", f_text)
            elif sen_type == 4:
                f_text = self.h_pattern.sub("[unused0] [unused7] [unused1]", f_text)
        else:
            f_text = self.h_pattern.sub("[unused0] "+p_head+" [unused1]", f_text)
        
        if t_flag:
            if sen_type == 1:
                f_text = self.t_pattern.sub("[unused2] [unused6] [unused3]", f_text)
            elif sen_type == 2:
                f_text = self.t_pattern.sub("[unused2] [unused7] [unused3]", f_text)
            elif sen_type == 3:
                f_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", f_text)
            elif sen_type == 4:
                f_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", f_text)
        else:
            f_text = self.t_pattern.sub("[unused2] "+p_tail+" [unused3]", f_text)

        f_text = f_text.split()

        try:
            h_pos = f_text.index("[unused0]")
            h_pos_l = f_text.index("[unused1]")
            t_pos = f_text.index("[unused2]") 
            t_pos_l = f_text.index("[unused3]")
            f_text.remove('[unused0]')
            f_text.remove('[unused1]')
            f_text.remove('[unused2]')
            f_text.remove('[unused3]')
            if h_pos < t_pos:
                h_pos_l -= 1
                t_pos -= 2
                t_pos_l -= 3
            else:
                t_pos_l -= 1
                h_pos -= 2
                h_pos_l -= 3
        except:
            print('wrong here!')
            h_pos = 1
            h_pos_l = 2
            t_pos = 1
            t_pos_l = 2
        
        tokenized_input = self.entityMarker.tokenizer.convert_tokens_to_ids(f_text)
        if sen_type != 1:
            tokenized_input = tokenized_input[1: ]
            h_pos -= 1
            t_pos -= 1
            h_pos_l -= 1
            t_pos_l -= 1
        if sen_type not in [3, 4]:
            tokenized_input = tokenized_input[: -1]
        return (tokenized_input, h_pos, t_pos, h_pos_l, t_pos_l, r)

    def __pos_pair_R__(self, scope, relation2sen_soft_t, ent2sen_tok, max_sample_num):
        pos_scope = list(range(scope[0], scope[1]))
        
        # shuffle bag to get different pairs
        all_pos_pair = []
        random.shuffle(relation2sen_soft_t)

        for i, item_1 in enumerate(relation2sen_soft_t):
            if i >= max_sample_num:
                break
            h_flag = random.random() > self.args.alpha
            t_flag = random.random() > self.args.alpha
            hop_flag = random.random() > self.args.alpha

            e1 = self.get_sen(random.choice(ent2sen_tok[item_1[0]]), h_flag, hop_flag, 1)
            e2 = self.get_sen(random.choice(ent2sen_tok[item_1[1]]), hop_flag, t_flag, 3)

            e_single = random.choice(pos_scope)
            while e_single in self.ignore_id:
                e_single = random.choice(pos_scope)
            
            all_pos_pair.append([('2', e1, e2), ('1', e_single)])

        return all_pos_pair

    def __pos_pair_doc__(self, scope_r1, scope_r2, scope_r3, relation2sen_soft_t, ent2sen_tok, max_sample_num):
        pos_scope_r1 = list(range(scope_r1[0], scope_r1[1]))
        pos_scope_r2 = list(range(scope_r2[0], scope_r2[1]))
        pos_scope_r3 = list(range(scope_r3[0], scope_r3[1]))
        
        # shuffle bag to get different pairs
        all_pos_pair = []
        random.shuffle(relation2sen_soft_t)

        for i, item_1 in enumerate(relation2sen_soft_t):
            if i >= max_sample_num:
                break
            h_flag = random.random() > self.args.alpha
            t_flag = random.random() > self.args.alpha
            hop_flag = random.random() > self.args.alpha

            sample_list_e1 = random.sample(ent2sen_tok[item_1[0]], 2)
            sample_list_e2 = random.sample(ent2sen_tok[item_1[0]], 2)
            e1 = self.get_sen(sample_list_e1[0], h_flag, hop_flag, 1)
            e2 = self.get_sen(sample_list_e2[0], hop_flag, t_flag, 3)
            e1_2 = self.get_sen(sample_list_e1[1], h_flag, hop_flag, 1)
            e2_2 = self.get_sen(sample_list_e2[1], hop_flag, t_flag, 3)

            e_single_r1 = random.choice(pos_scope)
            while e_single_r1 in self.ignore_id:
                e_single_r1 = random.choice(pos_scope)
            e_single_r2 = random.choice(pos_scope)
            while e_single_r2 in self.ignore_id:
                e_single_r2 = random.choice(pos_scope)
            e_single_r3 = random.choice(pos_scope)
            while e_single_r3 in self.ignore_id:
                e_single_r3 = random.choice(pos_scope)
            
            all_pos_pair.append([('2', e1, e2, e1_2, e2_2), ('1', e_single_r1, e_single_r2, e_single_r3)])

        return all_pos_pair

    def __pos_pair_R_q__(self, scope, relation2sen_soft_q, ent2sen_tok, max_sample_num):
        pos_scope = list(range(scope[0], scope[1]))
        
        all_pos_pair = []
        random.shuffle(relation2sen_soft_q)
        
        for i, item_1 in enumerate(relation2sen_soft_q):
            if i >= max_sample_num:
                break
            h_flag = random.random() > self.args.alpha
            t_flag = random.random() > self.args.alpha
            hop_flag_1 = random.random() > self.args.alpha
            hop_flag_2 = random.random() > self.args.alpha

            e1 = self.get_sen(random.choice(ent2sen_tok[item_1[0]]), h_flag, hop_flag_1, 1)
            e2 = self.get_sen(random.choice(ent2sen_tok[item_1[1]]), hop_flag_1, hop_flag_2, 2)
            e3 = self.get_sen(random.choice(ent2sen_tok[item_1[2]]), hop_flag_2, t_flag, 4)

            e_single = random.choice(pos_scope)
            while e_single in self.ignore_id:
                e_single = random.choice(pos_scope)
            all_pos_pair.append([('3', e1, e2, e3), ('1', e_single)])
        
        return all_pos_pair
    
    def __sample__(self):
        #single sentence
        self.pos_pair = []
        for rel in self.rel2scope.keys():
            scope = self.rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)

        #2-hop sentence
        self.pos_pair_R = []
        flag = 0
        mean_num = np.mean([len(self.relation2sen_soft_t[rel]) for rel in list(self.relation_map_t.keys())])
        max_sample_num = mean_num / 3
        print('max_sample_num_2-hop: ' + str(max_sample_num))
        rel_list_2_hop = list(self.relation_map_t.keys())
        random.shuffle(rel_list_2_hop)
        while len(self.pos_pair_R) < len(self.pos_pair):
            for rel in rel_list_2_hop:
                scope = self.rel2scope[self.relation_map_t[rel]]
                pos_pair_R = self.__pos_pair_R__(scope, self.relation2sen_soft_t[rel], self.ent2sen_tok, max_sample_num)
                self.pos_pair_R.extend(pos_pair_R)
            if flag == 0:
                flag = 1
                print('2-hop num:' + str(len(self.pos_pair_R)))
        self.pos_pair_R = self.pos_pair_R[: len(self.pos_pair)]
        random.shuffle(self.pos_pair_R)

        #3-hop sentence
        self.pos_pair_R_q = []
        flag = 0
        mean_num = np.mean([len(self.relation2sen_soft_q[rel]) for rel in list(self.relation_map_q.keys())])
        max_sample_num = mean_num / 30
        print('max_sample_num_3-hop: ' + str(max_sample_num))
        rel_list_3_hop = list(self.relation_map_q.keys())
        random.shuffle(rel_list_3_hop)
        while len(self.pos_pair_R_q) < len(self.pos_pair):
            for rel in rel_list_3_hop:
                scope = self.rel2scope[self.relation_map_q[rel]]
                pos_pair_R_q = self.__pos_pair_R_q__(scope, self.relation2sen_soft_q[rel], self.ent2sen_tok, max_sample_num)
                self.pos_pair_R_q.extend(pos_pair_R_q)
            if flag == 0:
                flag = 1
                print('3-hop hum:' + str(len(self.pos_pair_R_q)))
        self.pos_pair_R_q = self.pos_pair_R_q[: len(self.pos_pair)]
        random.shuffle(self.pos_pair_R_q)
        '''
        #doc loss
        self.pos_pair_doc = []
        mean_num = np.mean([len(self.relation2sen_soft_t[rel]) for rel in list(self.relation_map_t.keys())])
        max_sample_num = mean_num / 3
        print('max_sample_num_2-hop: ' + str(max_sample_num))
        rel_list_2_hop = list(self.relation_map_t.keys())
        random.shuffle(rel_list_2_hop)
        while len(self.pos_pair_doc) < len(self.pos_pair):
            for rel in rel_list_2_hop:
                scope_r1 = self.rel2scope[rel.split('#')[0]]
                scope_r2 = self.rel2scope[rel.split('#')[1]]
                scope_r3 = self.rel2scope[self.relation_map_t[rel]]
                pos_pair_doc = self.__pos_pair_doc__(scope_r1, scope_r2, scope_r3, self.relation2sen_soft_t[rel], self.ent2sen_tok, max_sample_num)
                self.pos_pair_doc.extend(pos_pair_doc)
        self.pos_pair_doc = self.pos_pair_doc[: len(self.pos_pair)]
        random.shuffle(self.pos_pair_doc)
        '''

        print("Postive pair's number is %d, Postive pair_R's number is %d" % (len(self.pos_pair), len(self.pos_pair_R)))

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        while(index in self.ignore_id):
            index = random.randint(0, len(self.pos_pair) - 1)
        bag = self.pos_pair[index]
        input = np.zeros((self.args.max_length * 2), dtype=int)
        mask = np.zeros((self.args.max_length * 2), dtype=int)
        label = np.zeros((2), dtype=int)
        h_pos = np.zeros((2), dtype=int)
        t_pos = np.zeros((2), dtype=int)
        h_pos_l = np.zeros((2), dtype=int)
        t_pos_l = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            if ind[0] == '1':
                ind = ind[1]
                input[i*self.args.max_length : (i+1)*self.args.max_length] = self.tokens[ind]
                mask[i*self.args.max_length : (i+1)*self.args.max_length] = self.mask[ind]
                label[i] = self.label[ind]
                h_pos[i] = self.h_pos[ind]
                t_pos[i] = self.t_pos[ind]
                h_pos_l[i] = self.h_pos_l[ind]
                t_pos_l[i] = self.t_pos_l[ind]
            else:
                assert False
        
        #2-hop sentences
        bag = self.pos_pair_R[index]
        input_R = np.zeros((self.args.max_length * 4), dtype=int)
        mask_R = np.zeros((self.args.max_length * 4), dtype=int)
        label_R = np.zeros((2), dtype=int)
        h_pos_R = np.zeros((2), dtype=int)
        t_pos_R = np.zeros((2), dtype=int)
        h_pos_l_R = np.zeros((2), dtype=int)
        t_pos_l_R = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            if ind[0] == '1':
                ind = ind[1]
                input_R[2*i*self.args.max_length : (2*i+1)*self.args.max_length] = self.tokens[ind]
                mask_R[2*i*self.args.max_length : (2*i+1)*self.args.max_length] = self.mask[ind]
                label_R[i] = self.label[ind]
                h_pos_R[i] = self.h_pos[ind]
                t_pos_R[i] = self.t_pos[ind]
                h_pos_l_R[i] = self.h_pos_l[ind]
                t_pos_l_R[i] = self.t_pos_l[ind]
            elif ind[0] == '2':
                #ids, ph, pt, sentence['r']
                ids_1, ph_1, pt_1, ph_l_1, pt_l_1, r_1 = ind[1]
                ids_2, ph_2, pt_2, ph_l_2, pt_l_2, r_2 = ind[2]
                ids = ids_1 + ids_2
                ph = min(self.args.max_length * 2 - 1, ph_1)
                pt = min(self.args.max_length * 2 - 1, len(ids_1) + pt_2)
                ph_l = min(self.args.max_length * 2, ph_l_1)
                pt_l = min(self.args.max_length * 2, len(ids_1) + pt_l_2)
                length = min(len(ids), self.args.max_length * 2)

                input_R[2*i*self.args.max_length : 2*i*self.args.max_length + length] = ids[: length]
                mask_R[2*i*self.args.max_length : 2*i*self.args.max_length + length] = 1

                label_R[i] = self.rel2id[self.relation_map_t['#'.join([r_1, r_2])]]
                h_pos_R[i] = ph
                t_pos_R[i] = pt
                h_pos_l_R[i] = ph_l
                t_pos_l_R[i] = pt_l
                
                assert ph < ph_l
                assert pt < pt_l
            else:
                assert False
        
        #3-hop sentences
        bag = self.pos_pair_R_q[index]
        input_R_q = np.zeros((self.args.max_length * 6), dtype=int)
        mask_R_q = np.zeros((self.args.max_length * 6), dtype=int)
        label_R_q = np.zeros((2), dtype=int)
        h_pos_R_q = np.zeros((2), dtype=int)
        t_pos_R_q = np.zeros((2), dtype=int)
        h_pos_l_R_q = np.zeros((2), dtype=int)
        t_pos_l_R_q = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            if ind[0] == '1':
                ind = ind[1]
                input_R_q[3*i*self.args.max_length : (3*i+1)*self.args.max_length] = self.tokens[ind]
                mask_R_q[3*i*self.args.max_length : (3*i+1)*self.args.max_length] = self.mask[ind]
                label_R_q[i] = self.label[ind]
                h_pos_R_q[i] = self.h_pos[ind]
                t_pos_R_q[i] = self.t_pos[ind]
                h_pos_l_R_q[i] = self.h_pos_l[ind]
                t_pos_l_R_q[i] = self.t_pos_l[ind]
                assert h_pos_R_q[i] < h_pos_l_R_q[i]
                assert t_pos_R_q[i] < t_pos_l_R_q[i]
            elif ind[0] == '3':
                ids_1, ph_1, pt_1, ph_l_1, pt_l_1, r_1 = ind[1]
                ids_2, ph_2, pt_2, ph_l_2, pt_l_2, r_2 = ind[2]
                ids_3, ph_3, pt_3, ph_l_3, pt_l_3, r_3 = ind[3]
                ids = ids_1 + ids_2 + ids_3
                ph = min(self.args.max_length * 3 - 1, ph_1)
                pt = min(self.args.max_length * 3 - 1, len(ids_1) + len(ids_2) + pt_3)
                ph_l = min(self.args.max_length * 3, ph_l_1)
                pt_l = min(self.args.max_length * 3, len(ids_1) + len(ids_2) + pt_l_3)
                assert ph < ph_l
                assert pt < pt_l
                length = min(len(ids), self.args.max_length * 3)

                input_R_q[3*i*self.args.max_length : 3*i*self.args.max_length + length] = ids[: length]
                mask_R_q[3*i*self.args.max_length : 3*i*self.args.max_length + length] = 1

                label_R_q[i] = self.rel2id[self.relation_map_q['#'.join([r_1, r_2, r_3])]]
                h_pos_R_q[i] = ph
                t_pos_R_q[i] = pt
                h_pos_l_R_q[i] = ph_l
                t_pos_l_R_q[i] = pt_l
                '''
                r2n = json.load(open('/data2/private/qinyujia/data_gen/zero_shot_data/relation_name_desc.json', 'r'))
                r2n = {k: v['name'] for k,v in r2n.items()}
                id2rel = {v:k for k,v in self.rel2id.items()}
                print(r2n[id2rel[label_R[i]]])
                print(' '.join([self.idx2token[x] for x in ids if x != 0]))
                print('\n')
                '''
            else:
                assert False

        #negative sampling
        index_neg = random.randint(0, len(self.pos_pair) - 1)
        while index_neg in self.ignore_id:
            index_neg = random.randint(0, len(self.pos_pair) - 1)
        
        bag = self.pos_pair[index_neg]
        input_R_neg = np.zeros((self.args.max_length * 6), dtype=int)
        mask_R_neg = np.zeros((self.args.max_length * 6), dtype=int)
        label_R_neg = np.zeros((2), dtype=int)
        h_pos_R_neg = np.zeros((2), dtype=int)
        t_pos_R_neg = np.zeros((2), dtype=int)
        h_pos_l_R_neg = np.zeros((2), dtype=int)
        t_pos_l_R_neg = np.zeros((2), dtype=int)

        for i, ind in enumerate(bag):
            if ind[0] == '1':
                ind = ind[1]
                input_R_neg[3*i*self.args.max_length : (3*i+1)*self.args.max_length] = self.tokens[ind]
                mask_R_neg[3*i*self.args.max_length : (3*i+1)*self.args.max_length] = self.mask[ind]
                label_R_neg[i] = self.label[ind]
                h_pos_R_neg[i] = self.h_pos[ind]
                t_pos_R_neg[i] = self.t_pos[ind]
                h_pos_l_R_neg[i] = self.h_pos_l[ind]
                t_pos_l_R_neg[i] = self.t_pos_l[ind]
                assert h_pos_R_neg[i] < h_pos_l_R_neg[i]
                assert t_pos_R_neg[i] < t_pos_l_R_neg[i]
            else:
                assert False
        

        single_sentence = (input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l)
        two_hop_pos = (input_R, mask_R, label_R, h_pos_R, t_pos_R, h_pos_l_R, t_pos_l_R)
        three_hop_pos = (input_R_q, mask_R_q, label_R_q, h_pos_R_q, t_pos_R_q, h_pos_l_R_q, t_pos_l_R_q)
        neg = (input_R_neg, mask_R_neg, label_R_neg, h_pos_R_neg, t_pos_R_neg, h_pos_l_R_neg, t_pos_l_R_neg)
        return single_sentence, two_hop_pos, three_hop_pos, neg

class MTBDataset(data.Dataset):
    """Overwritten class Dataset for model MTB.

    This class prepare data for training of MTB.
    """
    def __init__(self, path, args):
        """Inits tokenized sentence and positive pair for MTB.
        
        Args:
            path: path to your dataset.
            args: args from command line.
        
        Returns:
            No returns
        
        Raises:
            If the dataset in `path` is not the same format as described in 
            file 'prepare_data.py', there may raise:
                - `key nor found`
                - `integer can't be indexed`
                and so on.
        """
        self.path = path
        self.args = args
        data = json.load(open(os.path.join(path, "mtbdata.json")))
        entityMarker = EntityMarker()
        
        # Important Configures
        tot_sentence = len(data)

        # Converts tokens to ids and meanwhile `BLANK` some entities randomly.
        self.tokens = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.mask = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.h_pos = np.zeros((tot_sentence), dtype=int)
        self.t_pos = np.zeros((tot_sentence), dtype=int)
        self.h_pos_l = np.zeros((tot_sentence), dtype=int)
        self.t_pos_l = np.zeros((tot_sentence), dtype=int)
        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]  
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt, ph_l, pt_l = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag, 0)
            length = min(len(ids), args.max_length)
            self.tokens[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(args.max_length-1, ph)
            self.t_pos[i] = min(args.max_length-1, pt)
            self.h_pos_l[i] = min(args.max_length, ph_l)
            self.t_pos_l[i] = min(args.max_length, pt_l)
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)

        entpair2scope = json.load(open(os.path.join(path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(path, "entpair2negpair.json")))
        self.entpair2scope = entpair2scope
        self.entpair2negpair = entpair2negpair
        self.pos_pair = []
        
        # Generates all positive pair.
        for key in entpair2scope.keys():
            self.pos_pair.extend(self.__pos_pair__(entpair2scope[key]))
        print("Positive pairs' number is %d" % len(self.pos_pair))
        # Samples negative pairs dynamically.
        self.__sample__()

    def __sample__(self):    
        """Sample hard negative pairs.

        Sample hard negative pairs for MTB. As described in `prepare_data.py`, 
        `entpair2negpair` is ` A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). ` 
        
        ! IMPORTANT !
        We firstly get all hard negative pairs which may be a very huge number and then we sam
        ple negaitive pair where sampling number equals positive pairs' numebr. Using our 
        dataset, this code snippet can run normally. But if your own dataset is very big, this 
        code snippet will cost a lot of memory.
        """
        entpair2scope = self.entpair2scope
        entpair2negpair = self.entpair2negpair

        neg_pair = []

        # Gets all negative pairs.
        for key in entpair2negpair.keys():
            my_scope = entpair2scope[key]
            entpairs = entpair2negpair[key]
            if len(entpairs) == 0:
                continue
            for entpair in entpairs:
                neg_scope = entpair2scope[entpair]
                neg_pair.extend(self.__neg_pair__(my_scope, neg_scope))
        print("(MTB)Negative pairs number is %d" %len(neg_pair))
        
        # Samples a same number of negative pair with positive pairs. 
        random.shuffle(neg_pair)
        self.neg_pair = neg_pair[0:len(self.pos_pair)]
        del neg_pair # save the memory 

    def __pos_pair__(self, scope):
        """Gets all positive pairs.

        Args:
            scope: A scope in which all sentences share the same
                entity pair(head entity and tail entity).
        
        Returns:
            pos_pair: All positive pairs in a scope. The number of 
                positive pairs in a scope is (N-1)N/2 where N equals
                scope[1] - scope[0]
        """
        ent_scope = list(range(scope[0], scope[1]))
        pos_pair = []
        for i in range(len(ent_scope)):
            for j in range(i+1, len(ent_scope)):
                pos_pair.append([ent_scope[i], ent_scope[j]])
        return pos_pair

    def __neg_pair__(self, my_scope, neg_scope):
        """Gets all negative pairs in different scope.

        Args:
            my_scope: A scope which is samling negative pairs.
            neg_scope: A scope where sentences share only one entity
                with sentences in my_scope.
        
        Returns:
            neg_pair: All negative pair. Sentences in different scope 
                make up negative pairs.
        """
        my_scope = list(range(my_scope[0], my_scope[1]))
        neg_scope = list(range(neg_scope[0], neg_scope[1]))
        neg_pair = []
        for i in my_scope:
            for j in neg_scope:
                neg_pair.append([i, j])
        return neg_pair

    def __len__(self):
        """Number of instances in an epoch.
        
        Overwitten function.
        """
        return len(self.pos_pair)

    def __getitem__(self, index):
        """Gets training instance.

        If index is odd, we will return nagative instance, otherwise 
        positive instance. So in a batch, the number of positive pairs 
        equal the number of negative pairs.

        Args:
            index: Data index.
        
        Returns:
            {l,h}_input: Tokenized word id.
            {l,h}_mask: Attention mask for bert. 0 means masking, 1 means not masking.
            {l,h}_ph: Position of head entity.
            {l,h}_pt: Position of tail entity.
            label: Positive or negative.

            Setences in the same position in l_input and r_input is a sentence pair
            (positive or negative).
        """
        if index % 2 == 0:
            l_ind = self.pos_pair[index][0]
            r_ind = self.pos_pair[index][1]
            label = 1
        else:
            l_ind = self.neg_pair[index][0]
            r_ind = self.neg_pair[index][1]
            label = 0
        
        l_input = self.tokens[l_ind]
        l_mask = self.mask[l_ind]
        l_ph = self.h_pos[l_ind]
        l_pt = self.t_pos[l_ind]
        l_ph_l = self.h_pos_l[l_ind]
        l_pt_l = self.t_pos_l[l_ind]
        r_input = self.tokens[r_ind]
        r_mask = self.mask[r_ind]
        r_ph = self.h_pos[r_ind]
        r_pt = self.t_pos[r_ind]
        r_ph_l = self.h_pos_l[r_ind]
        r_pt_l = self.t_pos_l[r_ind]

        return l_input, l_mask, l_ph, l_pt, l_ph_l, l_pt_l, r_input, r_mask, r_ph, r_pt, r_ph_l, r_pt_l, label
