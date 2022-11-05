import os 
import pdb 
import torch
import torch.nn as nn 
from pytorch_metric_learning.losses.ntxent_loss import GenericPairLoss
from transformers import BertForMaskedLM, BertTokenizer, BertForPreTraining

class NTXentLoss(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0

class NTXentLoss_noise(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple

        pos_pair, neg_pair = [], []
        top_k_indices = torch.topk(mat[a1, p], int(a1.size()[0] * 0.7))[1]
        a1 = a1[top_k_indices]
        p = p[top_k_indices] 
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, (a1, p, a2, n))

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0

class NTXentLoss_R(GenericPairLoss):

    def __init__(self, temperature, args, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature
        self.args = args

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple

        mask1 = a1.le(self.args.batch_size_per_gpu-1)
        a1 = torch.masked_select(a1, mask1)
        p = torch.masked_select(p, mask1)

        mask2 = n.ge(self.args.batch_size_per_gpu)
        a2 = torch.masked_select(a2, mask2)
        n = torch.masked_select(n, mask2)

        a1_new = []
        p_new = []
        for i in range(a1.size()[0]):
            #if p[i] % 2 == 1 and a1[i] % 2 == 0:
            if p[i] - a1[i] == 1 and a1[i] % 2 == 0:
                a1_new.append(a1[i])
                p_new.append(p[i])
        a1 = torch.stack(a1_new, dim = 0)
        p = torch.stack(p_new, dim = 0)

        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        
        return self._compute_loss(pos_pair, neg_pair, (a1, p, a2, n))

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp), -log_exp
        return 0

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    
    Args:
        inputs: Inputs to mask. (batch_size, max_length) 
        tokenizer: Tokenizer.
        not_mask_pos: Using to forbid masking entity mentions. 1 for not mask.
    
    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

class CP(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP, self).__init__()
        #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if os.path.exists('/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'):
            load_path = '/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'
        else:
            load_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12'
        self.model = BertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.args = args
    
    def forward(self, input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * 2)
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)
        h_pos_l = h_pos_l.view(-1)
        t_pos_l = t_pos_l.view(-1)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1
        for i in range(input.size()[0]):
            not_mask_pos[i, h_pos[i]: h_pos_l[i]] = 1
            not_mask_pos[i, t_pos[i]: t_pos_l[i]] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[1]

        outputs = m_outputs

        # entity marker starter
        batch_size = input.size()[0]
        indice = torch.arange(0, batch_size)
        h_state = []
        t_state = []
        for i in range(batch_size):
            h_state.append(torch.mean(outputs[0][i, h_pos[i]: h_pos_l[i]], dim = 0))
            t_state.append(torch.mean(outputs[0][i, t_pos[i]: t_pos_l[i]], dim = 0))
        h_state = torch.stack(h_state, dim = 0)
        t_state = torch.stack(t_state, dim = 0)
        #h_state = outputs[0][indice, h_pos] # (batch_size * 2, hidden_size)
        #t_state = outputs[0][indice, t_pos]
        state = torch.cat((h_state, t_state), 1)
        r_loss = self.ntxloss(state, label)

        return m_loss, r_loss

class CP_noise(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP_noise, self).__init__()
        #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if os.path.exists('/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'):
            load_path = '/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'
        else:
            load_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12'
        self.model = BertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.ntxloss_noise = NTXentLoss_noise(temperature=args.temperature)
        self.args = args 
    
    def forward(self, input, mask, label, h_pos, t_pos):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * 2)
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[1]

        outputs = m_outputs

        # entity marker starter
        batch_size = input.size()[0]
        indice = torch.arange(0, batch_size)
        h_state = outputs[0][indice, h_pos] # (batch_size * 2, hidden_size)
        t_state = outputs[0][indice, t_pos]
        state = torch.cat((h_state, t_state), 1)
        r_loss = self.ntxloss_noise(state, label)

        return m_loss, r_loss

class CP_R(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP_R` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP_R, self).__init__()
        #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if os.path.exists('/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'):
            load_path = '/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'
        else:
            load_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12'
        self.model = BertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        #ckpt = torch.load('/data1/private/qinyujia/REAnalysis/ckpt/ckpt_cp_r/ckpt_of_step_5000')
        #self.model.bert.load_state_dict(ckpt["bert-base"])
        self.args = args 
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.ntxloss_R = NTXentLoss_R(temperature=args.temperature, args=self.args)
    
    def get_loss(self, input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l, loss_type):
        # masked language model loss
        if loss_type == 'CP':
            input = input.view(-1, self.args.max_length)
            mask = mask.view(-1, self.args.max_length)
        elif loss_type == '2-hop':
            input = input.view(-1, 2 * self.args.max_length)
            mask = mask.view(-1, 2 * self.args.max_length)
        elif loss_type == '3-hop':
            input = input.view(-1, 3 * self.args.max_length)
            mask = mask.view(-1, 3 * self.args.max_length)
            
        label = label.view(-1) # (batch_size * 2)
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)
        h_pos_l = h_pos_l.view(-1)
        t_pos_l = t_pos_l.view(-1)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        for i in range(input.size()[0]):
            not_mask_pos[i, h_pos[i]: h_pos_l[i]] = 1
            not_mask_pos[i, t_pos[i]: t_pos_l[i]] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[1]

        outputs = m_outputs

        # entity marker starter
        batch_size = input.size()[0]
        indice = torch.arange(0, batch_size)

        h_state = []
        t_state = []
        for i in range(batch_size):
            h_state.append(torch.mean(outputs[0][i, h_pos[i]: h_pos_l[i]], dim = 0))
            t_state.append(torch.mean(outputs[0][i, t_pos[i]: t_pos_l[i]], dim = 0))
        h_state = torch.stack(h_state, dim = 0)
        t_state = torch.stack(t_state, dim = 0)

        state = torch.cat((h_state, t_state), 1)
        
        if loss_type == 'CP':
            r_loss = self.ntxloss(state, label)
            return m_loss, r_loss

        elif loss_type in ['2-hop', '3-hop']:
            r_loss, debug = self.ntxloss_R(state, label)
            return m_loss, r_loss, debug

        #return m_loss, r_loss, debug

    def forward(self, input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l, input_R, mask_R, label_R, h_pos_R, t_pos_R, h_pos_l_R, t_pos_l_R, input_R_q, mask_R_q, label_R_q, h_pos_R_q, t_pos_R_q, h_pos_l_R_q, t_pos_l_R_q):
        m_loss, r_loss = self.get_loss(input, mask, label, h_pos, t_pos, h_pos_l, t_pos_l, 'CP')

        m_loss_R, r_loss_R, debug_2 = self.get_loss(input_R, mask_R, label_R, h_pos_R, t_pos_R, h_pos_l_R, t_pos_l_R, '2-hop')

        m_loss_R_q, r_loss_R_q, debug_3 = self.get_loss(input_R_q, mask_R_q, label_R_q, h_pos_R_q, t_pos_R_q, h_pos_l_R_q, t_pos_l_R_q, '3-hop')

        return m_loss, m_loss_R, r_loss, r_loss_R, m_loss_R_q, r_loss_R_q

class MTB(nn.Module):
    """Matching the Blanks.

    This class implements `MTB` model based on model `BertForMaskedLM`.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        bceloss: Binary Cross Entropy loss.
    """
    def __init__(self, args):
        super(MTB, self).__init__()
        if os.path.exists('/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'):
            load_path = '/data1/private/qinyujia/all_bert_models/uncased_L-12_H-768_A-12'
        else:
            load_path = '/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12'
        self.model = BertForMaskedLM.from_pretrained(load_path)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args
    

    def forward(self, l_input, l_mask, l_ph, l_pt, l_ph_l, l_pt_l, r_input, r_mask, r_ph, r_pt, r_ph_l, r_pt_l, label):
        # compute not mask entity marker
        indice = torch.arange(0, l_input.size()[0])
        l_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 
        r_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        l_not_mask_pos[indice, l_ph] = 1
        l_not_mask_pos[indice, l_pt] = 1

        r_not_mask_pos[indice, r_ph] = 1
        r_not_mask_pos[indice, r_pt] = 1

        # masked language model loss
        m_l_input, m_l_labels = mask_tokens(l_input.cpu(), self.tokenizer, l_not_mask_pos)
        m_r_input, m_r_labels = mask_tokens(r_input.cpu(), self.tokenizer, r_not_mask_pos) 
        m_l_outputs = self.model(input_ids=m_l_input, masked_lm_labels=m_l_labels, attention_mask=l_mask)
        m_r_outputs = self.model(input_ids=m_r_input, masked_lm_labels=m_r_labels, attention_mask=r_mask)
        m_loss = m_l_outputs[1] + m_r_outputs[1]

        # sentence pair relation loss 
        l_outputs = m_l_outputs
        r_outputs = m_r_outputs
        
        batch_size = l_input.size()[0]
        indice = torch.arange(0, batch_size)
        
        # left output
        l_h_state = []
        l_t_state = []
        for i in range(batch_size):
            l_h_state.append(torch.mean(l_outputs[0][i, l_ph[i]: l_ph_l[i]], dim = 0))
            l_t_state.append(torch.mean(l_outputs[0][i, l_pt[i]: l_pt_l[i]], dim = 0))
        l_h_state = torch.stack(l_h_state, dim = 0)
        l_t_state = torch.stack(l_t_state, dim = 0)
        l_state = torch.cat((l_h_state, l_t_state), 1) # (batch, 2 * hidden_size)
        
        # right output 
        r_h_state = []
        r_t_state = []
        for i in range(batch_size):
            r_h_state.append(torch.mean(r_outputs[0][i, r_ph[i]: r_ph_l[i]], dim = 0))
            r_t_state.append(torch.mean(r_outputs[0][i, r_pt[i]: r_pt_l[i]], dim = 0))
        r_h_state = torch.stack(r_h_state, dim = 0)
        r_t_state = torch.stack(r_t_state, dim = 0)
        r_state = torch.cat((r_h_state, r_t_state), 1) # (batch, 2 * hidden_size)

        # cal similarity
        similarity = torch.sum(l_state * r_state, 1) # (batch)

        # cal loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss 
