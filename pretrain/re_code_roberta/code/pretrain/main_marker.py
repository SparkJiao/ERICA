import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from apex import amp
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *


def log_loss(step_record, loss_record):
    if not os.path.exists("../../res"):
        os.mkdir("../../res")
    plt.plot(step_record, loss_record, lw=2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join("../../res", 'loss_curve.png'))
    plt.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args, model, train_dataset):
    # total step
    step_tot = (len(train_dataset)  // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    params = {"batch_size": args.batch_size_per_gpu, "sampler": train_sampler}
    train_dataloader = data.DataLoader(train_dataset, **params)
    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)
    print('111')
    # amp training
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    loss_record = []
    step_record = []
    for i in range(args.max_epoch):
        for step, batch in enumerate(train_dataloader):
            if args.model == "MTB":
                inputs = {"l_input": batch[0].to(args.device), "l_mask": batch[1].to(args.device), "l_ph": batch[2].to(args.device), "l_pt": batch[3].to(args.device), "r_input": batch[4].to(args.device), "r_mask": batch[5].to(args.device), "r_ph": batch[6].to(args.device),"r_pt": batch[7].to(args.device),"label": batch[8].to(args.device)}
            elif args.model == "CP":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device), "label": batch[2].to(args.device), "h_pos": batch[3].to(args.device), 't_pos': batch[4].to(args.device)}
            elif args.model == "CP_noise":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device), "label": batch[2].to(args.device), "h_pos": batch[3].to(args.device), 't_pos': batch[4].to(args.device)}
            elif args.model == "CP_large":
                inputs = {"input": batch[0].to(args.device), "mask": batch[1].to(args.device), "label": batch[2].to(args.device), "h_pos": batch[3].to(args.device), 't_pos': batch[4].to(args.device)}
            elif args.model == "CP_R":
                batch_1 = batch[0]
                batch_2 = batch[1]
                batch_3 = batch[2]
                batch_4 = batch[3]

                def cat_data(batch_x, batch_y, hop_type):
                    if hop_type == '2-hop':
                        neg_batch = batch_y.to(args.device)
                        neg_batch = torch.cat([neg_batch[0: int(args.batch_size_per_gpu / 2), 0: 2 * args.max_length], neg_batch[0: int(args.batch_size_per_gpu / 2), 3 * args.max_length: 5 * args.max_length]], dim = 1)
                        return torch.cat([batch_x.to(args.device)[: int(args.batch_size_per_gpu / 2)], neg_batch], dim = 0)
                    elif hop_type == '3-hop':
                        return torch.cat([batch_x.to(args.device)[: int(args.batch_size_per_gpu / 2)], batch_y.to(args.device)[int(args.batch_size_per_gpu / 2): ]], dim = 0)
                    else:
                        assert False
                 
                input_R = cat_data(batch_2[0], batch_4[0], '2-hop')
                mask_R = cat_data(batch_2[1], batch_4[1], '2-hop')
                label_R = cat_data(batch_2[2], batch_4[2], '2-hop')
                h_pos_R = cat_data(batch_2[3], batch_4[3], '2-hop')
                t_pos_R = cat_data(batch_2[4], batch_4[4], '2-hop')

                input_R_q = cat_data(batch_3[0], batch_4[0], '3-hop')
                mask_R_q = cat_data(batch_3[1], batch_4[1], '3-hop')
                label_R_q = cat_data(batch_3[2], batch_4[2], '3-hop')
                h_pos_R_q = cat_data(batch_3[3], batch_4[3], '3-hop')
                t_pos_R_q = cat_data(batch_3[4], batch_4[4], '3-hop')

                #input_R = torch.cat([batch_2[0].to(args.device)[: args.batch_size_per_gpu / 2], batch_3[0].to(args.device)[args.batch_size_per_gpu / 2: , 0: 2 * args.max_length]], dim = 0)

                inputs = {"input": batch_1[0].to(args.device), "mask": batch_1[1].to(args.device), "label": batch_1[2].to(args.device), "h_pos": batch_1[3].to(args.device), 't_pos': batch_1[4].to(args.device),
                "input_R": input_R, "mask_R": mask_R, "label_R": label_R, "h_pos_R": h_pos_R, 't_pos_R': t_pos_R, "input_R_q": input_R_q, "mask_R_q": mask_R_q, "label_R_q": label_R_q, "h_pos_R_q": h_pos_R_q, 't_pos_R_q': t_pos_R_q}
            
            model.train()

            if args.model == "CP_R":
                m_loss, m_loss_R, r_loss, r_loss_R, m_loss_R_q, r_loss_R_q = model(**inputs)

                loss = (m_loss + m_loss_R + r_loss + r_loss_R + m_loss_R_q + r_loss_R_q) / 3
            else:
                m_loss, r_loss = model(**inputs)

                loss = m_loss + r_loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            if step % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    if not os.path.exists("../../ckpt"):
                        os.mkdir("../../ckpt")
                    if not os.path.exists("../../ckpt/"+args.save_dir):
                        os.mkdir("../../ckpt/"+args.save_dir)
                    ckpt = {
                        'bert-base': model.module.model.bert.state_dict(),
                    }
                    torch.save(ckpt, os.path.join("../../ckpt/"+args.save_dir, "ckpt_of_step_"+str(global_step)))

                # if args.local_rank in [0, -1] and global_step % 5 == 0:
                #     step_record.append(global_step)
                #     loss_record.append(loss)
                
                # if args.local_rank in [0, -1] and global_step % 500 == 0:
                #     log_loss(step_record, loss_record)
                
                if args.local_rank in [0, -1]:
                    if args.model == "CP_R":
                        sys.stdout.write("step: %d, shcedule: %.3f, mlm_loss: %.6f mlm_loss_R: %.6f mlm_loss_R_q: %.6f relation_loss: %.6f relation_loss_R: %.6f relation_loss_R_q: %.6f\r" % (global_step, global_step/step_tot, m_loss, m_loss_R, m_loss_R_q, r_loss, r_loss_R, r_loss_R_q))
                    else:
                        sys.stdout.write("step: %d, shcedule: %.3f, mlm_loss: %.6f relation_loss: %.6f\r" % (global_step, global_step/step_tot, m_loss, r_loss))
                    sys.stdout.flush()
        
        if args.train_sample:
            print("sampling...")
            train_dataloader.dataset.__sample__()
            print("sampled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, 
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=32, help="batch size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch number")
    
    parser.add_argument("--alpha", dest="alpha", type=float,
                        default=0.3, help="true entity(not `BLANK`) proportion")

    parser.add_argument("--model", dest="model", type=str,
                        default="", help="{MTB, CP}")
    parser.add_argument("--train_sample",action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=2, help="bag size")
    parser.add_argument("--temperature", dest="temperature", type=float,
                        default=0.05, help="temperature for NTXent loss")
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768, help="hidden size for mlp")
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    
    parser.add_argument("--save_step", dest="save_step", type=int, 
                        default=10000, help="step to save")
    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="", help="ckpt dir to save")
    

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")

    parser.add_argument("--n_gpu", dest="n_gpu", type=int,
                        default=4, help="n_gpu")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    # print args
    print(args)
    # set cuda 
    if not os.path.exists("/mnt/yardcephfs/mmyard/g_wxg_td_prc/yujiaqin"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    set_seed(args)
    
    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("../../log"):
            os.mkdir("../../log")
        with open("../../log/pretrain_log", 'a+') as f:
            f.write(str(time.ctime())+"\n")
            f.write(str(args)+"\n")
            f.write("----------------------------------------------------------------------------\n")

    # Model and dataset
    if args.model == "MTB":
        model = MTB(args).to(args.device)
        if not os.path.isfile('train_dataset_MTB'):
            train_dataset = MTBDataset("../../data/MTB", args)
            torch.save(train_dataset, 'train_dataset_MTB')
        else:
            train_dataset = torch.load('train_dataset_MTB')
    elif args.model == "CP":
        model = CP(args).to(args.device)
        print('preparing data')
        if not os.path.isfile('train_dataset_CP'):
            train_dataset = CPDataset("../../data/CP", args)
            torch.save(train_dataset, 'train_dataset_CP')
        else:
            train_dataset = torch.load('train_dataset_CP')
    elif args.model == "CP_R":
        print('preparing data')
        if args.debug == 0:
            if not os.path.isfile('train_dataset_CP_R'):
                #if True:
                train_dataset = CP_R_Dataset("../../data/CP_R", args)
                torch.save(train_dataset, 'train_dataset_CP_R')
            else:
                train_dataset = torch.load('train_dataset_CP_R')
                print('loading end')
        elif args.debug == 1:
            if not os.path.isfile('train_dataset_CP_R_debug'):
                #if True:
                train_dataset = CP_R_Dataset("../../data/CP_R", args)
                torch.save(train_dataset, 'train_dataset_CP_R_debug')
            else:
                train_dataset = torch.load('train_dataset_CP_R_debug')
        #exit()
        print('preparing end')
        model = CP_R(args).to(args.device)
    elif args.model == "CP_noise":
        model = CP_noise(args).to(args.device)
        print('preparing data')
        if not os.path.isfile('train_dataset_CP'):
            train_dataset = CPDataset("../../data/CP", args)
            torch.save(train_dataset, 'train_dataset_CP')
        else:
            train_dataset = torch.load('train_dataset_CP')
    elif args.model == "CP_large":
        model = CP(args).to(args.device)
        print('preparing data')
        if not os.path.isfile('train_dataset_CP_large'):
            train_dataset = CPDataset("../../data/CP_R_large", args)
            torch.save(train_dataset, 'train_dataset_CP_large')
        else:
            train_dataset = torch.load('train_dataset_CP_large')
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {MTB, CP}")

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    train(args, model, train_dataset)
