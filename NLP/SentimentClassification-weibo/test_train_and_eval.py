import json
import logging
import os, sys
import time
import numpy as no

from tqdm import tqdm, trange
import torch
from utils import collate_batch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup
from net.utils.fgm import FGM
from net.utils.data_gen import DataGen


def trains(args, train_dataset, eval_dataset, model, fold_num=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=args.collate_batch)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight', 'transitions']
    bert_params = ['bert.embeddings', 'bert.encoder']
    # named_parameters迭代打印每一次迭代元素的名字和param
    # no_decay列表参数都不在模型参数中，保留参数值，采用参数权重衰减系数；存在也保留参数值，把权重衰减系数置为0。
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if
                                                (not any(nd in n for nd in no_decay)) and any(
                                                    nr in n for nr in bert_params)], 'weight_decay': args.weight_decay},
                                    {'params': [p for n, p in model.named_parameters() if
                                                any(nd in n for nd in no_decay) and any(nr in n for nr in bert_params)],
                                     'weight_decay': 0.0},
                                    {'params': [p for n, p in model.named_parameters() if
                                                (not any(nd in n for nd in no_decay)) and (
                                                    not any(nr in n for nr in bert_params))],
                                     'weight_decay': args.weight_decay},
                                    {'params': [p for n, p in model.named_parameters() if
                                                any(nd in n for nd in no_decay) and \
                                                (not any(nr in n for nr in bert_params))], 'weight_decay': 0.0}, ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_rate * t_total,num_training_steps=t_total)
    # lambda1 = lambda epoch: float(epoch >= 2)
    # lambda2 = lambda epoch: 1.0
    # bert_lr_scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda1, lambda2, lambda2])
    logging.info('*'*35)
    logging.info("***** Running training *****")
    logging.info("  Device = %s", args.device)
    logging.info("  Model name = %s", str(args.__dict__))
    logging.info("  Learning rate = %s", str(args.learning_rate))
    logging.info("  Warmup rate = %s", str(args.warmup_rate))
    logging.info("  Weight Decay = %s", str(args.weight_decay))
    logging.info("  label smooth = %s", str(args.label_smooth))
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d",
                 args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss,logging_loss = 0.0,0.0
    model.zero_grad()
    optimizer.step()

    best_f_score = 0.
    best_epoch = 0

    if args.attack == 'fgm':
        fgm = FGM(model)
        logging.info('*** attack method = fgm ***')
    elif args.attack == 'de':
        dg = DataGen(model)
        logging.info('*** attack method = gen ***')

    for epoch in range(args.num_train_epochs):
        logging.info('  Epoch [{}/{}]'.format(epoch + 1,args.num_train_epochs))
        # 调整学习率，前2轮bert部分为0
        # bert_lr_scheduler.step()
        # print(bert_lr_scheduler.get_last_lr())
        for step,batch in enumerate(train_dataloader):
            model.train()
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(args.device)
            outputs = model(**inputs)
            loss,logits = outputs[0],outputs[1]
            # logging.info('*** loss = %f ***',loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            logging_loss += loss.item()
            tr_loss += loss.item()

            if args.attack == 'fgm':
                # logger.info("*****do attack*****")
                # fgm 攻击
                fgm.attack()
                outputs = model(**inputs)
                loss_adv,_logits = outputs[0],outputs[1]
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数
                # fgm 攻击 end
            elif args.attack == 'de':
                # # dg attack
                # dg.attack()
                # outputs = model(**inputs)
                # loss_adv, _logits = outputs[0], outputs[1]
                # loss_adv.backward()
                # dg.restore()

                emd_name = 'word_embedding'
                iters =2
                xi = 10
                epsilon =1
                inputs['original_logits'] = logits.detach().data.clone()
                for name, param in model.named_parameters():
                    if param.requires_grad and emd_name in name:
                        delta1, delta2 = 0.0,torch.rand

