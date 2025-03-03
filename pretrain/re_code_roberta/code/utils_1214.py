import os 
import re
import pdb
import ast 
import json
import random
import argparse
import numpy as np
import pandas as pd 
from tqdm import trange
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict, Counter

class EntityMarker():
    """Converts raw text to BERT-input ids and finds entity position.

    Attributes:
        tokenizer: Bert-base tokenizer.
        h_pattern: A regular expression pattern -- * h *. Using to replace head entity mention.
        t_pattern: A regular expression pattern -- ^ t ^. Using to replace tail entity mention.
        err: Records the number of sentences where we can't find head/tail entity normally.
        args: Args from command line. 
    """
    def __init__(self, args=None):
        if os.path.exists('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin'):
            load_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12'
            self.tokenizer_roberta = RobertaTokenizer.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/roberta-base')
        else:
            load_path = '/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'
            self.tokenizer_roberta = RobertaTokenizer.from_pretrained('/data1/private/qinyujia/all_bert_models/roberta-base')
        self.args = args
        if args.model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(load_path)
        elif args.model == 'roberta':
            self.tokenizer = self.tokenizer_roberta
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args
    
    def tokenize(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False, single = True):
        if self.args.model == 'bert':
            return self.tokenize_bert(raw_text, h_pos_li, t_pos_li, h_type, t_type, h_blank, t_blank, single)
        elif self.args.model == 'roberta':
            return self.tokenize_roberta(raw_text, h_pos_li, t_pos_li, h_type, t_type, h_blank, t_blank, single)

    def tokenize_roberta(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False, single = True):
        h_mention = ' '.join(raw_text[h_pos_li[0]: h_pos_li[-1]])
        t_mention = ' '.join(raw_text[t_pos_li[0]: t_pos_li[-1]])
        
        tokens = []
        for i, token in enumerate(raw_text):
            if i >= h_pos_li[0] and i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['<s>'] + raw_text[h_pos_li[0]: h_pos_li[-1]] + ['</s>']
                continue
            if i >= t_pos_li[0] and i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['<s>'] + raw_text[t_pos_li[0]: t_pos_li[-1]] + ['</s>']
                continue
            tokens.append(token)

        # tokenize
        tokenized_text = [self.tokenizer.cls_token] + self.tokenizer.tokenize(' '.join(tokens)) + ['</s>']
        
        i = 1
        pos = []
        h_pos = 0
        t_pos = 0
        h_pos_l = 0
        t_pos_l = 0
        flag_1 = 0
        flag_2 = 0

        while (i <= len(tokenized_text) - 2):
            if tokenized_text[i] == '<s>':
                xx = i + 1
                while (tokenized_text[i] != '</s>'):
                    i += 1
                yy = i
                i += 1
                pos.append((xx, yy))
            else:
                i += 1
        assert len(pos) == 2
        if h_pos_li[0] < t_pos_li[0]:
            h_pos = pos[0][0]
            h_pos_l = pos[0][1]
            t_pos = pos[1][0]
            t_pos_l = pos[1][1]
        else:
            t_pos = pos[0][0]
            t_pos_l = pos[0][1]
            h_pos = pos[1][0]
            h_pos_l = pos[1][1]

        tokenized_input = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        
        return tokenized_input, h_pos, t_pos, h_pos_l, t_pos_l

    def tokenize_bert(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False, single = True):
        tokens = []
        h_mention = [] 
        t_mention = []
        for i, token in enumerate(raw_text):
            token = token.lower()    
            if i >= h_pos_li[0] and i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_li[0] and i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        # tokenize
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_head = self.tokenizer.tokenize(h_mention)
        tokenized_tail = self.tokenizer.tokenize(t_mention)

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        # If head entity type and tail entity type are't None, 
        # we use `CT` settings to tokenize raw text, i.e. replacing 
        # entity mention with entity type.
        if h_type != None and t_type != None:
            p_head = h_type
            p_tail = t_type

        if not single:
            f_text = "[CLS] " + p_text + " [SEP]"
            return f_text, p_head, p_tail

        if h_blank:
            p_text = self.h_pattern.sub("[unused0] [unused4] [unused1]", p_text)
        else:
            p_text = self.h_pattern.sub("[unused0] "+p_head+" [unused1]", p_text)
        if t_blank:
            p_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", p_text)
        else:
            p_text = self.t_pattern.sub("[unused2] "+p_tail+" [unused3]", p_text)

        
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        
        # If h_pos_li and t_pos_li overlap, we can't find head entity or tail entity.

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
            self.err += 1
            h_pos = 0
            h_pos_l = 1
            t_pos = 0
            t_pos_l = 1

        tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        
        return tokenized_input, h_pos, t_pos, h_pos_l, t_pos_l
 
    def tokenize_OMOT(self, head, tail, h_first):
        """Tokenizer for `CM` and `CT` settings.

        This function converts head entity and tail entity to ids.

        Args:
            head: Head entity(mention or type). Please ensure that this argument has
                been tokenized using bert tokenizer.
            tail: Tail entity(mention or type). Please ensure that this argument has
                been tokenized using bert tokenizer.
            h_first: Whether head entity is the first entity(i.e. head entity in 
            original sentence is in front of tail entity),
        
        Returns:
            tokenized_input: Input ids that can be the input to BERT directly.
            h_pos: Head entity position(head entity marker start positon).
            t_pos: Tail entity position(tail entity marker start positon).
        """
        tokens = ['[CLS]',]
        if h_first:
            h_pos = 1
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]
            t_pos = len(tokens)
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]
            
        else:
            t_pos = 1
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]
            h_pos = len(tokens)
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]

        tokens.append('[SEP]')
        tokenized_input = self.tokenizer.convert_tokens_to_ids(tokens)

        return tokenized_input, h_pos, t_pos

def sample_trainset(dataset, prop):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)
    
    little_data = []
    reduced_times = 1 / prop
    rel2ins = defaultdict(list)
    for ins in data:
        rel2ins[ins['relation']].append(ins)
    for key in rel2ins.keys():
        sens = rel2ins[key]
        random.shuffle(sens)
        number = int(len(sens) // reduced_times) if len(sens) % reduced_times == 0 else int(len(sens) // reduced_times) + 1
        little_data.extend(sens[:number])
    print("We sample %d instances in "+dataset+" train set." % len(little_data))
    
    f = open(dataset+"/train_" + str(prop) + ".txt",'w')
    for ins in little_data:
        text = json.dumps(ins)
        f.write(text + '\n')
    f.close()

def get_type2id(dataset):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)

    type2id = {'UNK':0}
    for ins in data:
        if 'subj_'+ins['h']['type'] not in type2id:
            type2id['subj_'+ins['h']['type']] = len(type2id)
            type2id['obj_'+ins['h']['type']] = len(type2id)
        if 'subj_'+ins['t']['type'] not in type2id:
            type2id['subj_'+ins['t']['type']] = len(type2id)
            type2id['obj_'+ins['t']['type']] = len(type2id)

    json.dump(type2id, open(dataset+"/type2id.json", 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="dataset")
    args = parser.parse_args()

    sample_trainset(os.path.join('../data', args.dataset), 0.01)
    sample_trainset(os.path.join('../data', args.dataset), 0.1)
    # get_type2id(os.path.join('../data', args.dataset))     


    


