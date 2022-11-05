import torch
import pdb 
import torch.nn as nn
from transformers import BertModel, RobertaModel


class REModel(nn.Module):
    """relation extraction model
    """
    def __init__(self, args, weight=None):
        super(REModel, self).__init__()
        self.args = args 
        self.training = True
        
        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        scale = 2 if args.entity_marker else 1
        self.rel_fc = nn.Linear(args.hidden_size*scale, args.rel_num)
        if args.model == 'bert':
            self.bert = BertModel.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
            if args.ckpt_to_load != "None":
                print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
                ckpt = torch.load("/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/REAnalysis_doc_new/ckpt/"+args.ckpt_to_load)
                self.bert.load_state_dict(ckpt["bert-base"])
            else:
                print("*******No ckpt to load, Let's use bert base!*******")
        elif args.model == 'roberta':
            self.bert = RobertaModel.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/roberta-base')
            if args.ckpt_to_load != "None":
                print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
                ckpt = torch.load("/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/REAnalysis_doc_new/ckpt/"+args.ckpt_to_load)
                self.bert.load_state_dict(ckpt["bert-base"])
            else:
                print("*******No ckpt to load, Let's use bert base!*******")
        
    def forward(self, input_ids, mask, h_pos, t_pos, label, h_pos_l, t_pos_l):
        # bert encode
        outputs = self.bert(input_ids, mask)

        # entity marker
        if self.args.entity_marker:
            indice = torch.arange(input_ids.size()[0])
            #h_state = outputs[0][indice, h_pos]
            #t_state = outputs[0][indice, t_pos]

            h_state = []
            t_state = []
            for i in range(input_ids.size()[0]):
                h_state.append(torch.mean(outputs[0][i, h_pos[i]: h_pos_l[i]], dim = 0))
                t_state.append(torch.mean(outputs[0][i, t_pos[i]: t_pos_l[i]], dim = 0))
            h_state = torch.stack(h_state, dim = 0)
            t_state = torch.stack(t_state, dim = 0)

            state = torch.cat((h_state, t_state), 1) #(batch_size, hidden_size*2)
        else:
            #[CLS]
            state = outputs[0][:, 0, :] #(batch_size, hidden_size)

        # linear map
        logits = self.rel_fc(state) #(batch_size, rel_num)
        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output    
        






