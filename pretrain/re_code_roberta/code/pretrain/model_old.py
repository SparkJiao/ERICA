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

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        mask1 = a1.le(15)
        a1 = torch.masked_select(a1, mask1)
        p = torch.masked_select(p, mask1)

        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        mask1 = a1.le(15)
        a1 = torch.masked_select(a1, mask1)

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
        self.model = BertForMaskedLM.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
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
        self.model = BertForMaskedLM.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
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
        self.model = BertForMaskedLM.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin/all_bert_models/uncased_L-12_H-768_A-12')
        #ckpt = torch.load('/data1/private/qinyujia/REAnalysis/ckpt/ckpt_cp_r/ckpt_of_step_5000')
        #self.model.bert.load_state_dict(ckpt["bert-base"])
        self.args = args
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.ntxloss_R = NTXentLoss_R(temperature=args.temperature)

    def get_loss(self, input, mask, label, h_pos, t_pos, loss_type):
        # masked language model loss
        if loss_type == 'CP':
            input = input.view(-1, self.args.max_length)
            mask = mask.view(-1, self.args.max_length)
        elif loss_type == 'R':
            input = input.view(-1, 2 * self.args.max_length)
            mask = mask.view(-1, 2 * self.args.max_length)
            '''
            print('111')
            print(input[0])
            print(mask[0])
            print(label[0])
            print(h_pos[0])
            print(t_pos[0])
            print('222')
            print(input[1])
            print(mask[1])
            print(label[1])
            print(h_pos[1])
            print(t_pos[1])
            '''
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
        if loss_type == 'CP':
            r_loss = self.ntxloss(state, label)
        elif loss_type == 'R':
            r_loss = self.ntxloss_R(state, label)

        return m_loss, r_loss

    def forward(self, input, mask, label, h_pos, t_pos, input_R, mask_R, label_R, h_pos_R, t_pos_R):
        m_loss, r_loss = self.get_loss(input, mask, label, h_pos, t_pos, 'CP')

        m_loss_R, r_loss_R = self.get_loss(input_R, mask_R, label_R, h_pos_R, t_pos_R, 'R')


        return m_loss, m_loss_R, r_loss, r_loss_R

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
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args


    def forward(self, l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label):
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
        l_h_state = l_outputs[0][indice, l_ph] # (batch, hidden_size)
        l_t_state = l_outputs[0][indice, l_pt] # (batch, hidden_size)
        l_state = torch.cat((l_h_state, l_t_state), 1) # (batch, 2 * hidden_size)

        # right output
        r_h_state = r_outputs[0][indice, r_ph]
        r_t_state = r_outputs[0][indice, r_pt]
        r_state = torch.cat((r_h_state, r_t_state), 1)

        # cal similarity
        similarity = torch.sum(l_state * r_state, 1) # (batch)

        # cal loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss
