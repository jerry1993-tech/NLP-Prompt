# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : train.py
# @Description: UIE fine-tune
# @Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/finetune.py


import argparse
import os
import time
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from metrics import SpanEvaluator
from processor import convert_example
from utils import download_pretrained_model, set_seed
from train_logger import SummaryWriter

# 指定运行设备
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1024, type=int, help="Random seed for initialization")
parser.add_argument("--pretrained_model", default='uie-base-chinese', type=str,
                    choices=['uie-base-chinese'], help="backbone of encoder.")
parser.add_argument("--train_path", default='data/simple_ner/train.txt', type=str, help="The path of train set.")
parser.add_argument("--dev_path", default='data/simple_ner/dev.txt', type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default='./checkpoint', type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=300, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequency.")
parser.add_argument('--device', default='{}'.format(device), type=str, choices=['cpu', 'gpu'],
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--log_name", default='log.png', type=str, help="log image name.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
args = parser.parse_args()

writer = SummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def do_train():
	set_seed(args.seed)
	if not os.path.exists(args.pretrained_model):
		download_pretrained_model(args.pretrained_model)
	model = torch.load(os.path.join(args.pretrained_model, 'pytorch_model.bin'),
	                   map_location=args.device)                                          # 加载预训练好的UIE模型，模型结构见：model.UIE()
	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)                      # 加载 ernie3 的tokenizer
	# 加载数据
	dataset = load_dataset('text', data_files={'train': args.train_path,
	                                           'dev': args.dev_path})
	print(dataset)
	convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
	dataset = dataset.map(convert_func, batched=True)

	train_dataset = dataset["train"]
	eval_dataset = dataset["dev"]
	train_dataloader = DataLoader(train_dataset, shuffle=True,
	                              collate_fn=default_data_collator,
	                              batch_size=args.batch_size)
	eval_dataloader = DataLoader(eval_dataset,
	                             collate_fn=default_data_collator,
	                             batch_size=args.batch_size)
	# 定义优化器
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
	model.to(args.device)

	# 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
	num_update_steps_per_epoch = len(train_dataloader)
	max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	warm_steps = int(args.warmup_ratio * max_train_steps)
	lr_scheduler = get_scheduler(
		name='linear',
		optimizer=optimizer,
		num_warmup_steps=warm_steps,
		num_training_steps=max_train_steps)

	loss_list = []
	tic_train = time.time()
	criterion = torch.nn.BCELoss()
	global_step, best_f1 = 0, 0
	metric = SpanEvaluator()

	for epoch in range(1, args.num_train_epochs + 1):
		for batch in train_dataloader:
			start_prob, end_prob = model(
				input_ids=batch['input_ids'].to(args.device),
				token_type_ids=batch['token_type_ids'].to(args.device),
				attention_mask=batch['attention_mask'].to(args.device)
			)
			start_ids = batch['start_ids'].to(torch.float32).to(args.device)  # (batch, seq_len)
			end_ids = batch['end_ids'].to(torch.float32).to(args.device)      # (batch, seq_len)

			loss_start = criterion(start_prob, start_ids)                     # 起止向量loss -> (1,)
			loss_end = criterion(end_prob, end_ids)                           # 结束向量loss -> (1,)
			loss = (loss_start + loss_end) / 2.0                              # 求平均 -> (1,)
			loss.backward()        # 反向传播
			optimizer.step()       # 梯度更新
			lr_scheduler.step()
			optimizer.zero_grad()  # 清除梯度
			loss_list.append(float(loss.cpu().detach()))

			global_step += 1
			if global_step % args.logging_steps == 0:
				time_diff = time.time() - tic_train
				loss_avg = sum(loss_list) / len(loss_list)
				writer.add_scalar('train_loss', loss_avg, global_step)
				print("global step: %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
				      % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
				tic_train = time.time()

			if global_step % args.valid_steps == 0:
				cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
				if not os.path.exists(cur_save_dir):
					os.makedirs(cur_save_dir)
				torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
				tokenizer.save_pretrained(cur_save_dir)

				precision, recall, f1 = evaluate(model, metric, eval_dataloader, global_step)
				print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))

				if f1 > best_f1:
					print(f"best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}")
					best_f1 = f1

					cur_save_dir = os.path.join(args.save_dir, "model_best")
					if not os.path.exists(cur_save_dir):
						os.makedirs(cur_save_dir)
					torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
					tokenizer.save_pretrained(cur_save_dir)
				tic_train = time.time()


def evaluate(model, metric, data_loader, global_step):
	"""
	在验证集上评估当前模型的训练效果
	:param model: 当前模型
	:param metric: 评估指标类(metric)
	:param data_loader: 验证集的 dataloader
	:param global_step: 当前的训练步数
	"""
	model.eval()
	metric.reset()

	for batch in data_loader:
		start_prob, end_prob = model(
			input_ids=batch['input_ids'].to(args.device),
			token_type_ids=batch['token_type_ids'].to(args.device),
			attention_mask=batch['attention_mask'].to(args.device)
		)
		start_ids = batch['start_ids'].to(torch.float32).detach().numpy()
		end_ids = batch['end_ids'].to(torch.float32).detach().numpy()
		num_correct, num_infer, num_label = metric.compute(
			start_prob.detach().cpu().numpy(),
			end_prob.detach().cpu().numpy(),
			start_ids,
			end_ids
		)
		metric.update(num_correct, num_infer, num_label)

	precision, recall, f1 = metric.accumulate()
	writer.add_scalar('eval-precision', precision, global_step)
	writer.add_scalar('eval-recall', recall, global_step)
	writer.add_scalar('eval-f1', f1, global_step)
	writer.record()

	model.train()

	return precision, recall, f1


if __name__ == '__main__':
	from rich import print
	do_train()

