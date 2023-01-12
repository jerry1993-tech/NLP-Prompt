# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : processor.py
# @Description: 主要是模型预处理/后处理函数
# @Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/paddlenlp/taskflow/utils.py#L810

import json

import numpy as np
from typing import List

import torch


def get_bool_ids_greater_than(probs: list, limit=0.5, return_prob=False) -> list:
	"""
	对概率值大于阈值的 token_ids 进行筛选
	:param probs: (_type_)
	:param limit: (float, optional),默认值为 0.5
	:param return_prob: (bool, optional),默认值为 False
	:return: list: [1, 3, 5, ...] (return_prob=False)
	or [[(1, 0.90), (3, 0.66), ...]] (return_prob=True)
	"""
	probs = np.array(probs)
	dim_len = len(probs.shape)
	if dim_len > 1:
		result = []
		for p in probs:
			result.append(get_bool_ids_greater_than(p, limit, return_prob))
		return result
	else:
		result = []
		for i, p in enumerate(probs):
			if p > limit:
				if return_prob:
					result.append((i, p))
				else:
					result.append(i)
		return result


def get_span(start_ids: list, end_ids: list, with_prob=False) -> set:
	"""
	根据输入 start_ids 和 end_ids，计算entity span列表
	:param start_ids: (list) [1, 2, 10] or [(1, 0.90), (2, 0.71), (10, 0.88)]
	:param end_ids: (list) [4, 12] or [(4, 0.87), (12, 0.61)]
	:param with_prob: (bool, optional), 默认值为 False
	:return: (set) set((2, 4), (10, 12))
	"""
	if with_prob:
		start_ids = sorted(start_ids, key=lambda x: x[0])
		end_ids = sorted(end_ids, key=lambda x: x[0])
	else:
		start_ids = sorted(start_ids)
		end_ids = sorted(end_ids)

	start_pointer = 0
	end_pointer = 0
	len_start = len(start_ids)
	len_end = len(end_ids)
	couple_dict = {}
	while start_pointer < len_start and end_pointer < len_end:
		if with_prob:
			if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
				couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
				start_pointer += 1
				end_pointer += 1
				continue
			if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
				couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
				start_pointer += 1
				continue
			if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
				end_pointer += 1
				continue
		else:
			if start_ids[start_pointer] == end_ids[end_pointer]:
				couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
				start_pointer += 1
				end_pointer += 1
				continue
			if start_ids[start_pointer] < end_ids[end_pointer]:
				couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
				start_pointer += 1
				continue
			if start_ids[start_pointer] > end_ids[end_pointer]:
				end_pointer += 1
				continue
	result = [(couple_dict[end], end) for end in couple_dict]
	result = set(result)
	return result


def convert_inputs(tokenizer, prompts: List[str], contents: List[str], max_seq_len=512) -> dict:
	"""
	处理输入样本，包括 prompt/content 的拼接和offset的计算
	:param tokenizer: transformers.AutoTokenizer
	:param prompts: prompt 文本列表
	:param contents: content 文本列表
	:param max_seq_len: 句子最大长度
	:return: dict -> {
                    'input_ids': tensor([[1, 57, 405, ...]]),
                    'token_type_ids': tensor([[0, 0, 0, ...]]),
                    'attention_mask': tensor([[1, 1, 1, ...]]),
                    'pos_ids': tensor([[0, 1, 2, 3, 4, 5,...]])
                    'offset_mapping': tensor([[[0, 0], [0, 1], [1, 2], [0, 0], [3, 4], ...]])
            }
	"""
	inputs = tokenizer(text=prompts,                 # [SEP]前内容
	                   text_pair=contents,           # [SEP]后内容
	                   truncation=True,              # 是否截断
	                   max_length=max_seq_len,       # 句子最大长度
	                   padding="max_length",         # padding 类型
	                   return_offsets_mapping=True   # 返回offsets用于计算token_id到原文的映射,为每一个单词相应的长度
	                   )
	pos_ids = []
	for i in range(len(contents)):                   # pos_ids 只针对 [SEP]后内容
		pos_ids += [[j for j in range(len(inputs['input_ids'][i]))]]
	pos_ids = torch.tensor(pos_ids)
	inputs['pos_ids'] = pos_ids

	offset_mappings = [[list(x) for x in offset] for offset in inputs["offset_mapping"]]
	"""
	Desc:
        经过text & text_pair后，生成的offset_mapping会将prompt和content的offset独立计算，
        这里需将content的offset位置补回去。
    
    Example:
        offset_mapping(before):[[0, 0], [0, 1], [1, 2], [2, 3], [0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], ...]
        offset_mapping(after):[[0, 0], [0, 1], [1, 2], [2, 3], [0, 0], [4, 5], [5, 6], ...]
	"""
	for i in range(len(offset_mappings)):  # offset 重计算
		bias = 0
		for index in range(1, len(offset_mappings[i])):
			mapping = offset_mappings[i][index]
			if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
				bias = offset_mappings[i][index - 1][1]
			if mapping[0] == 0 and mapping[1] == 0:
				continue
			offset_mappings[i][index][0] += bias
			offset_mappings[i][index][1] += bias

	inputs['offset_mapping'] = offset_mappings

	for k, v in inputs.items():
		inputs[k] = torch.LongTensor(v)                      # list转tensor

	return inputs


def map_offset(ori_offset, offset_mapping):
	"""
	把 ori offset计算真实的 token的offset
	"""
	for index, span in enumerate(offset_mapping):
		if span[0] <= ori_offset < span[1]:
			return index
	return -1


def convert_example(examples, tokenizer, max_seq_len) -> dict:
	"""
	将训练样本数据转成模型输入的形式
	:param tokenizer: transformers.AutoTokenizer
	:param examples: (dict) 训练数据样本 -> {
												"text": [
			                                                '{
			                                                    "content": "昨天北京飞上海话费一百元",
			                                                    "prompt": "时间",
			                                                    "result_list": [{"text": "昨天", "start": 0, "end": 2}]
			                                                    }',
			                                                ...
			                                                ]
			                                    }
	:param max_seq_len: 句子最大长度
	:return: dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'pos_ids': [[0, 1, 2, ...], [0, 1, 2, ...], ...],
                            'start_ids': [[0, 1, 0, ...], [0, 0, ..., 1, ...]],
                            'end_ids': [[0, 0, 1, ...], [0, 0, ...]]
                        }
	"""
	tokenized_output = {
		'input_ids': [],
		'token_type_ids': [],
		'attention_mask': [],
		'pos_ids': [],
		'start_ids': [],
		'end_ids': []
	}

	for example in examples['text']:
		example = json.loads(example)                # json转化成字典 <dict> = json.loads(<json>)
		try:
			encoded_inputs = tokenizer(
				text=example['prompt'],
				text_pair=example['content'],
				stride=len(example['prompt']),        # 定义附加token的数量,溢出token将包含 prompt 中的token
				truncation=True,
				max_length=max_seq_len,
				padding='max_length',
				return_offsets_mapping=True
			)
		except:
			print('Warning! ERROR Sample: ', example)
			exit()
		offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
		"""
		Desc:
	        经过text & text_pair后，生成的offset_mapping会将prompt和content的offset独立计算，
	        这里需将content的offset位置补回去。

	    Example:
	        offset_mapping(before):[[0, 0], [0, 1], [1, 2], [2, 3], [0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], ...]
	        offset_mapping(after):[[0, 0], [0, 1], [1, 2], [2, 3], [0, 0], [4, 5], [5, 6], ...]
		"""
		bias = 0
		for index in range(len(offset_mapping)):
			if index == 0:
				continue
			mapping = offset_mapping[index]
			if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
				bias = index
			if mapping[0] == 0 and mapping[1] == 0:
				continue
			offset_mapping[index][0] += bias
			offset_mapping[index][1] += bias

		start_ids = [0 for x in range(max_seq_len)]
		end_ids = [0 for x in range(max_seq_len)]

		for item in example["result_list"]:
			start = map_offset(item["start"] + bias, offset_mapping)    # 计算真实的start token的id
			end = map_offset(item["end"] - 1 + bias, offset_mapping)    # 计算真实的end token的id
			start_ids[start] = 1.0                                      # 以 one-hot vector 表示
			end_ids[end] = 1.0                                          # 以 one-hot vector 表示

		pos_ids = [i for i in range(len(encoded_inputs['input_ids']))]
		tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
		tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
		tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
		tokenized_output['pos_ids'].append(pos_ids)
		tokenized_output['start_ids'].append(start_ids)
		tokenized_output['end_ids'].append(end_ids)

	for k, v in tokenized_output.items():
		tokenized_output[k] = np.array(v, dtype='int64')

	return tokenized_output


if __name__ == '__main__':

	probs = [0.12, 0.90, 0.2, 0.66, 0.45]
	result = get_bool_ids_greater_than(probs, return_prob=True)
	print(result)

	start_ids = [1, 2, 10]
	end_ids = [4, 12]
	result = get_span(start_ids, end_ids)
	print(result)

	from transformers import AutoTokenizer
	pretrained_model_path = './uie-base-chinese'
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)  # 加载tokenizer，ERNIE 3.0
	prompts = ["出发地", "时间"]
	contents = ["昨天北京飞上海话费一百元", "昨天北京飞上海话费一百元"]
	inputs = convert_inputs(tokenizer, prompts, contents)
	print("inputs=", inputs)

	example = {
				"text": [
                            '{"content": "昨天北京飞上海话费一百元", "prompt": "出发地", "result_list": [{"text": "北京", "start": 2, "end": 4}]}'
                            ]
                }
	tokenized_output = convert_example(example, tokenizer, 20)
	print(tokenized_output)
